import os

os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import random
import subprocess
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import transformers

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

DEVICE = "cuda"
DTYPE = torch.float16

SEED = 42

PROMPT_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048
]

MAX_NEW_TOKENS = 128

RUNS_PER_CONDITION = 3

WARMUP_RUNS = 2

COOLDOWN_SECONDS = 60

GPU_SAMPLE_INTERVAL = 1.0


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_RESULTS_FILE = (
    "smollm3_3b_context_sweep_runs.csv"
)

GPU_SAMPLES_FILE = (
    "smollm3_3b_context_sweep_gpu_samples.csv"
)

SUMMARY_FILE = (
    "smollm3_3b_context_sweep_summary.csv"
)

PAIRED_FILE = (
    "smollm3_3b_context_sweep_paired.csv"
)

ENVIRONMENT_FILE = (
    "smollm3_3b_context_sweep_environment.txt"
)


# ============================================================
# CONTROLLED BASE TEXT
# ============================================================

BASE_TEXT = """
Artificial intelligence has become an important component of
modern computer systems. It enables machines to process data,
recognize patterns, make predictions, automate tasks, support
decision making, understand language, analyze images, and solve
complex computational problems. Artificial intelligence is used
in healthcare, education, transportation, cybersecurity,
manufacturing, finance, robotics, scientific research, and many
other areas. Modern artificial intelligence systems rely on
algorithms, data, computing resources, and increasingly large
neural network models to perform complex tasks efficiently.
"""


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CUDA CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")


# ============================================================
# BASIC HELPERS
# ============================================================

def bytes_to_gib(value):
    return value / (1024 ** 3)


def bytes_to_mib(value):
    return value / (1024 ** 2)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================
# COOLDOWN
# ============================================================

def cooldown(seconds):

    print()

    for _ in tqdm(
        range(seconds),
        desc="Cooling GPU",
        unit="sec",
        ncols=80
    ):
        time.sleep(1)

    print("Cooldown complete.")


# ============================================================
# NVIDIA-SMI
# ============================================================

def get_nvidia_smi_stats():

    fields = [
        "timestamp",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "pstate"
    ]

    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits"
    ]

    try:

        output = subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()

        line = output.splitlines()[0]

        values = [
            value.strip()
            for value in line.split(",")
        ]

        return dict(zip(fields, values))

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# PYTORCH MEMORY STATS
# ============================================================

def get_torch_memory_stats():

    torch.cuda.synchronize()

    return {

        "torch_allocated_GiB":
            bytes_to_gib(
                torch.cuda.memory_allocated()
            ),

        "torch_reserved_GiB":
            bytes_to_gib(
                torch.cuda.memory_reserved()
            ),

        "torch_max_allocated_GiB":
            bytes_to_gib(
                torch.cuda.max_memory_allocated()
            ),

        "torch_max_reserved_GiB":
            bytes_to_gib(
                torch.cuda.max_memory_reserved()
            )
    }


# ============================================================
# GPU MONITOR
# ============================================================

class GPUMonitor:

    def __init__(
        self,
        run_number,
        context_length,
        pair_number,
        use_cache,
        interval=1.0
    ):

        self.run_number = run_number
        self.context_length = context_length
        self.pair_number = pair_number
        self.use_cache = use_cache
        self.interval = interval

        self.samples = []

        self.stop_event = threading.Event()

        self.thread = None


    def sample_once(self, stage):

        gpu = get_nvidia_smi_stats()

        torch_stats = get_torch_memory_stats()

        sample = {

            "Run":
                self.run_number,

            "Context_Length":
                self.context_length,

            "Pair":
                self.pair_number,

            "KV_Cache":
                self.use_cache,

            "Stage":
                stage,

            "Local_Time":
                datetime.now().isoformat()
        }


        if "error" not in gpu:

            sample.update(
                {

                    "GPU_Temperature_C":
                        safe_float(
                            gpu.get(
                                "temperature.gpu"
                            )
                        ),

                    "GPU_Utilization_percent":
                        safe_float(
                            gpu.get(
                                "utilization.gpu"
                            )
                        ),

                    "Memory_Utilization_percent":
                        safe_float(
                            gpu.get(
                                "utilization.memory"
                            )
                        ),

                    "NVIDIA_Memory_Used_MiB":
                        safe_float(
                            gpu.get(
                                "memory.used"
                            )
                        ),

                    "NVIDIA_Memory_Total_MiB":
                        safe_float(
                            gpu.get(
                                "memory.total"
                            )
                        ),

                    "SM_Clock_MHz":
                        safe_float(
                            gpu.get(
                                "clocks.current.sm"
                            )
                        ),

                    "Memory_Clock_MHz":
                        safe_float(
                            gpu.get(
                                "clocks.current.memory"
                            )
                        ),

                    "Power_Draw_W":
                        safe_float(
                            gpu.get(
                                "power.draw"
                            )
                        ),

                    "PState":
                        gpu.get(
                            "pstate"
                        )
                }
            )

        else:

            sample[
                "NVIDIA_SMI_Error"
            ] = gpu["error"]


        sample.update(torch_stats)

        self.samples.append(sample)


    def _monitor_loop(self):

        while not self.stop_event.is_set():

            self.sample_once(
                stage="during"
            )

            self.stop_event.wait(
                self.interval
            )


    def start(self):

        self.sample_once(
            stage="before"
        )

        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )

        self.thread.start()


    def stop(self):

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        self.sample_once(
            stage="after"
        )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print()

print("=" * 80)

print(
    "SMOLLM3-3B FP16 KV-CACHE CONTEXT-LENGTH SWEEP"
)

print("=" * 80)


gpu_properties = (
    torch.cuda.get_device_properties(0)
)


physical_gpu_memory_gib = (
    gpu_properties.total_memory
    /
    (1024 ** 3)
)


physical_gpu_memory_mib = (
    gpu_properties.total_memory
    /
    (1024 ** 2)
)


print(
    "Python:",
    sys.version.split()[0]
)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "Transformers:",
    transformers.__version__
)

print(
    "CUDA runtime:",
    torch.version.cuda
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    f"Physical GPU memory: "
    f"{physical_gpu_memory_gib:.4f} GiB"
)

print(
    f"Prompt lengths: "
    f"{PROMPT_LENGTHS}"
)

print(
    f"Generated tokens/run: "
    f"{MAX_NEW_TOKENS}"
)

print(
    f"Runs per condition: "
    f"{RUNS_PER_CONDITION}"
)


# ============================================================
# NVIDIA-SMI BEFORE MODEL LOAD
# ============================================================

initial_gpu_status = (
    get_nvidia_smi_stats()
)


print()

print("=" * 80)

print(
    "NVIDIA-SMI BEFORE MODEL LOAD"
)

print("=" * 80)


for key, value in (
    initial_gpu_status.items()
):
    print(
        f"{key}: {value}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()

print("=" * 80)

print(
    "LOADING TOKENIZER"
)

print("=" * 80)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    "Tokenizer loaded."
)


# ============================================================
# LOAD MODEL
# ============================================================

print()

print("=" * 80)

print(
    "LOADING MODEL"
)

print("=" * 80)


torch.cuda.empty_cache()


model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
        device_map="cuda",
        local_files_only=True
    )
)


model.eval()

torch.cuda.synchronize()


print(
    "Model loaded successfully."
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

config = model.config


num_layers = getattr(
    config,
    "num_hidden_layers",
    None
)


hidden_size = getattr(
    config,
    "hidden_size",
    None
)


num_attention_heads = getattr(
    config,
    "num_attention_heads",
    None
)


num_kv_heads = getattr(
    config,
    "num_key_value_heads",
    None
)


head_dimension = (
    hidden_size
    //
    num_attention_heads
)


print()

print("=" * 80)

print(
    "MODEL ARCHITECTURE"
)

print("=" * 80)


print(
    "Layers:",
    num_layers
)

print(
    "Hidden size:",
    hidden_size
)

print(
    "Attention heads:",
    num_attention_heads
)

print(
    "KV heads:",
    num_kv_heads
)

print(
    "Head dimension:",
    head_dimension
)


# ============================================================
# MODEL MEMORY
# ============================================================

parameter_bytes = sum(

    parameter.numel()
    *
    parameter.element_size()

    for parameter
    in model.parameters()
)


parameter_gib = (
    bytes_to_gib(
        parameter_bytes
    )
)


print(
    f"Parameter memory: "
    f"{parameter_gib:.4f} GiB"
)


if hasattr(
    model,
    "get_memory_footprint"
):

    hf_memory_bytes = (
        model.get_memory_footprint()
    )

    hf_memory_gib = (
        bytes_to_gib(
            hf_memory_bytes
        )
    )

else:

    hf_memory_bytes = None
    hf_memory_gib = None


if hf_memory_gib is not None:

    print(
        f"HF model footprint: "
        f"{hf_memory_gib:.4f} GiB"
    )


# ============================================================
# NVIDIA-SMI AFTER MODEL LOAD
# ============================================================

gpu_after_model = (
    get_nvidia_smi_stats()
)


print()

print(
    "NVIDIA-SMI after model load:"
)


for key, value in (
    gpu_after_model.items()
):

    print(
        f"{key}: {value}"
    )


# ============================================================
# KV CACHE THEORY
# ============================================================

BYTES_PER_ELEMENT = 2


kv_bytes_per_token = (

    2
    *
    num_layers
    *
    num_kv_heads
    *
    head_dimension
    *
    BYTES_PER_ELEMENT
)


print()

print("=" * 80)

print(
    "THEORETICAL KV CACHE"
)

print("=" * 80)


print(
    f"KV bytes/token: "
    f"{kv_bytes_per_token:,}"
)

print(
    f"KV KiB/token: "
    f"{kv_bytes_per_token / 1024:.3f}"
)


for context_length in PROMPT_LENGTHS:

    prompt_kv_mib = (

        context_length
        *
        kv_bytes_per_token
        /
        (1024 ** 2)
    )


    final_kv_mib = (

        (
            context_length
            +
            MAX_NEW_TOKENS
        )
        *
        kv_bytes_per_token
        /
        (1024 ** 2)
    )


    print(
        f"Context {context_length:4d}: "
        f"prompt KV = "
        f"{prompt_kv_mib:.3f} MiB, "
        f"final KV = "
        f"{final_kv_mib:.3f} MiB"
    )


# ============================================================
# BUILD EXACT-LENGTH TOKEN INPUT
# ============================================================

def create_exact_token_prompt(
    target_length
):

    base_ids = tokenizer.encode(
        BASE_TEXT,
        add_special_tokens=False
    )


    if len(base_ids) == 0:

        raise RuntimeError(
            "Base text produced no tokens."
        )


    repeats = (

        target_length
        //
        len(base_ids)

        +
        2
    )


    token_ids = (
        base_ids
        *
        repeats
    )


    token_ids = (
        token_ids[
            :target_length
        ]
    )


    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


    attention_mask = (
        torch.ones_like(
            input_ids
        )
    )


    assert (
        input_ids.shape[1]
        ==
        target_length
    )


    return (
        input_ids,
        attention_mask
    )


# ============================================================
# VERIFY EXACT PROMPT LENGTHS
# ============================================================

print()

print("=" * 80)

print(
    "VERIFYING PROMPT LENGTHS"
)

print("=" * 80)


for target_length in PROMPT_LENGTHS:

    test_ids, test_mask = (
        create_exact_token_prompt(
            target_length
        )
    )


    print(
        f"Requested: "
        f"{target_length}"
        f" | Actual: "
        f"{test_ids.shape[1]}"
    )


    del test_ids
    del test_mask


torch.cuda.empty_cache()


# ============================================================
# SINGLE BENCHMARK RUN
# ============================================================

def benchmark_once(
    context_length,
    use_cache,
    run_number,
    pair_number
):

    input_ids, attention_mask = (
        create_exact_token_prompt(
            context_length
        )
    )


    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()


    baseline_allocated = (
        torch.cuda.memory_allocated()
    )


    baseline_reserved = (
        torch.cuda.memory_reserved()
    )


    monitor = GPUMonitor(
        run_number=run_number,
        context_length=context_length,
        pair_number=pair_number,
        use_cache=use_cache,
        interval=GPU_SAMPLE_INTERVAL
    )


    monitor.start()

    time.sleep(0.2)


    torch.cuda.synchronize()


    start_time = (
        time.perf_counter()
    )


    with torch.inference_mode():

        output_ids = model.generate(

            input_ids=input_ids,

            attention_mask=attention_mask,

            max_new_tokens=
                MAX_NEW_TOKENS,

            min_new_tokens=
                MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=use_cache,

            pad_token_id=
                tokenizer.pad_token_id
        )


    torch.cuda.synchronize()


    elapsed_time = (

        time.perf_counter()
        -
        start_time
    )


    monitor.stop()


    generated_tokens = (

        output_ids.shape[1]
        -
        context_length
    )


    tokens_per_second = (

        generated_tokens
        /
        elapsed_time
    )


    peak_allocated = (
        torch.cuda.max_memory_allocated()
    )


    peak_reserved = (
        torch.cuda.max_memory_reserved()
    )


    incremental_peak_allocated = (

        peak_allocated
        -
        baseline_allocated
    )


    theoretical_prompt_kv_mib = (

        context_length
        *
        kv_bytes_per_token
        /
        (1024 ** 2)
    )


    theoretical_final_kv_mib = (

        (
            context_length
            +
            MAX_NEW_TOKENS
        )
        *
        kv_bytes_per_token
        /
        (1024 ** 2)
    )


    result = {

        "Run":
            run_number,

        "Context_Length":
            context_length,

        "Pair":
            pair_number,

        "KV_Cache":
            use_cache,

        "Prompt_Tokens":
            context_length,

        "Generated_Tokens":
            generated_tokens,

        "Time_s":
            elapsed_time,

        "Tokens_per_sec":
            tokens_per_second,

        "Baseline_Allocated_GiB":
            bytes_to_gib(
                baseline_allocated
            ),

        "Baseline_Reserved_GiB":
            bytes_to_gib(
                baseline_reserved
            ),

        "Peak_Allocated_GiB":
            bytes_to_gib(
                peak_allocated
            ),

        "Peak_Reserved_GiB":
            bytes_to_gib(
                peak_reserved
            ),

        "Incremental_Peak_Allocated_MiB":
            bytes_to_mib(
                incremental_peak_allocated
            ),

        "Theoretical_Prompt_KV_MiB":
            theoretical_prompt_kv_mib,

        "Theoretical_Final_KV_MiB":
            theoretical_final_kv_mib
    }


    sample_df = (
        pd.DataFrame(
            monitor.samples
        )
    )


    if (
        "Stage"
        in sample_df.columns
    ):

        during = sample_df[
            sample_df["Stage"]
            ==
            "during"
        ]

    else:

        during = (
            pd.DataFrame()
        )


    mappings = {

        "GPU_Temperature_C":
            "GPU_Temp",

        "GPU_Utilization_percent":
            "GPU_Util",

        "Memory_Utilization_percent":
            "Memory_Util",

        "NVIDIA_Memory_Used_MiB":
            "Physical_VRAM",

        "SM_Clock_MHz":
            "SM_Clock",

        "Memory_Clock_MHz":
            "Memory_Clock",

        "Power_Draw_W":
            "Power"
    }


    if len(during) > 0:

        for source, name in (
            mappings.items()
        ):

            if source in during.columns:

                values = pd.to_numeric(
                    during[source],
                    errors="coerce"
                )


                result[
                    f"Mean_{name}"
                ] = values.mean()


                result[
                    f"Max_{name}"
                ] = values.max()


                result[
                    f"Min_{name}"
                ] = values.min()


    del output_ids
    del input_ids
    del attention_mask


    torch.cuda.empty_cache()


    return (
        result,
        monitor.samples
    )


# ============================================================
# WARM-UP
# ============================================================

print()

print("=" * 80)

print(
    "WARM-UP"
)

print("=" * 80)


WARMUP_CONTEXT = 256


for warmup_index in range(
    WARMUP_RUNS
):

    use_cache = bool(
        warmup_index
        %
        2
    )


    input_ids, attention_mask = (
        create_exact_token_prompt(
            WARMUP_CONTEXT
        )
    )


    print()

    print(
        f"Warm-up "
        f"{warmup_index + 1}/"
        f"{WARMUP_RUNS}"
        f" | context="
        f"{WARMUP_CONTEXT}"
        f" | cache="
        f"{use_cache}"
    )


    torch.cuda.synchronize()


    with torch.inference_mode():

        warmup_output = model.generate(

            input_ids=input_ids,

            attention_mask=attention_mask,

            max_new_tokens=
                MAX_NEW_TOKENS,

            min_new_tokens=
                MAX_NEW_TOKENS,

            do_sample=False,

            use_cache=use_cache,

            pad_token_id=
                tokenizer.pad_token_id
        )


    torch.cuda.synchronize()


    del warmup_output
    del input_ids
    del attention_mask


    torch.cuda.empty_cache()


    print(
        "Warm-up complete."
    )


    cooldown(
        COOLDOWN_SECONDS
    )


# ============================================================
# BUILD RANDOMIZED PAIRED SCHEDULE
# ============================================================

rng = random.Random(
    SEED
)


schedule = []


pair_id = 0


for context_length in (
    PROMPT_LENGTHS
):

    for repeat in range(
        1,
        RUNS_PER_CONDITION + 1
    ):

        pair_id += 1


        conditions = [
            False,
            True
        ]


        rng.shuffle(
            conditions
        )


        for use_cache in conditions:

            schedule.append(
                {

                    "Context_Length":
                        context_length,

                    "Pair":
                        pair_id,

                    "Repeat":
                        repeat,

                    "KV_Cache":
                        use_cache
                }
            )


print()

print("=" * 80)

print(
    "RANDOMIZED PAIRED SCHEDULE"
)

print("=" * 80)


for index, experiment in enumerate(
    schedule,
    start=1
):

    print(
        f"{index:02d}: "
        f"context="
        f"{experiment['Context_Length']:4d}"
        f" | pair="
        f"{experiment['Pair']:02d}"
        f" | repeat="
        f"{experiment['Repeat']}"
        f" | cache="
        f"{experiment['KV_Cache']}"
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

all_gpu_samples = []


print()

print("=" * 80)

print(
    "STARTING CONTEXT-LENGTH SWEEP"
)

print("=" * 80)


for run_number, experiment in enumerate(
    schedule,
    start=1
):

    context_length = (
        experiment[
            "Context_Length"
        ]
    )


    pair_number = (
        experiment[
            "Pair"
        ]
    )


    use_cache = (
        experiment[
            "KV_Cache"
        ]
    )


    repeat = (
        experiment[
            "Repeat"
        ]
    )


    print()

    print("-" * 80)

    print(
        f"Run "
        f"{run_number}/"
        f"{len(schedule)}"
    )

    print(
        f"Context length: "
        f"{context_length}"
    )

    print(
        f"Repeat: "
        f"{repeat}"
    )

    print(
        f"Pair: "
        f"{pair_number}"
    )

    print(
        f"KV cache: "
        f"{use_cache}"
    )

    print("-" * 80)


    result, gpu_samples = (
        benchmark_once(
            context_length=
                context_length,

            use_cache=
                use_cache,

            run_number=
                run_number,

            pair_number=
                pair_number
        )
    )


    result[
        "Repeat"
    ] = repeat


    results.append(
        result
    )


    all_gpu_samples.extend(
        gpu_samples
    )


    print()

    print(
        f"Runtime: "
        f"{result['Time_s']:.4f} s"
    )


    print(
        f"Throughput: "
        f"{result['Tokens_per_sec']:.3f} tok/s"
    )


    print(
        f"Peak allocated: "
        f"{result['Peak_Allocated_GiB']:.4f} GiB"
    )


    print(
        f"Peak reserved: "
        f"{result['Peak_Reserved_GiB']:.4f} GiB"
    )


    print(
        f"Incremental peak: "
        f"{result['Incremental_Peak_Allocated_MiB']:.2f} MiB"
    )


    if (
        "Max_Physical_VRAM"
        in result
    ):

        print(
            f"Max physical VRAM: "
            f"{result['Max_Physical_VRAM']:.0f} MiB"
        )


    if (
        "Mean_GPU_Util"
        in result
    ):

        print(
            f"Mean GPU utilization: "
            f"{result['Mean_GPU_Util']:.1f}%"
        )


    if (
        "Mean_Power"
        in result
    ):

        print(
            f"Mean power: "
            f"{result['Mean_Power']:.1f} W"
        )


    # --------------------------------------------------------
    # SAVE PARTIAL DATA
    # --------------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        RAW_RESULTS_FILE,
        index=False
    )


    pd.DataFrame(
        all_gpu_samples
    ).to_csv(
        GPU_SAMPLES_FILE,
        index=False
    )


    print(
        "Partial results saved."
    )


    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    if (
        run_number
        <
        len(schedule)
    ):

        cooldown(
            COOLDOWN_SECONDS
        )


# ============================================================
# FINAL DATAFRAMES
# ============================================================

results_df = (
    pd.DataFrame(
        results
    )
)


gpu_samples_df = (
    pd.DataFrame(
        all_gpu_samples
    )
)


# ============================================================
# SUMMARY BY CONTEXT AND CACHE CONDITION
# ============================================================

summary_rows = []


for context_length in (
    PROMPT_LENGTHS
):

    for use_cache in [
        False,
        True
    ]:

        subset = results_df[

            (
                results_df[
                    "Context_Length"
                ]
                ==
                context_length
            )

            &

            (
                results_df[
                    "KV_Cache"
                ]
                ==
                use_cache
            )
        ]


        summary_rows.append(
            {

                "Context_Length":
                    context_length,

                "KV_Cache":
                    use_cache,

                "N":
                    len(subset),

                "Mean_Time_s":
                    subset[
                        "Time_s"
                    ].mean(),

                "Median_Time_s":
                    subset[
                        "Time_s"
                    ].median(),

                "SD_Time_s":
                    subset[
                        "Time_s"
                    ].std(
                        ddof=1
                    ),

                "Min_Time_s":
                    subset[
                        "Time_s"
                    ].min(),

                "Max_Time_s":
                    subset[
                        "Time_s"
                    ].max(),

                "Mean_Tokens_per_sec":
                    subset[
                        "Tokens_per_sec"
                    ].mean(),

                "Median_Tokens_per_sec":
                    subset[
                        "Tokens_per_sec"
                    ].median(),

                "SD_Tokens_per_sec":
                    subset[
                        "Tokens_per_sec"
                    ].std(
                        ddof=1
                    ),

                "Mean_Peak_Allocated_GiB":
                    subset[
                        "Peak_Allocated_GiB"
                    ].mean(),

                "Mean_Peak_Reserved_GiB":
                    subset[
                        "Peak_Reserved_GiB"
                    ].mean(),

                "Mean_Incremental_Peak_MiB":
                    subset[
                        "Incremental_Peak_Allocated_MiB"
                    ].mean(),

                "Mean_Physical_VRAM_MiB":
                    subset[
                        "Mean_Physical_VRAM"
                    ].mean()
                    if
                    "Mean_Physical_VRAM"
                    in subset.columns
                    else
                    np.nan,

                "Max_Physical_VRAM_MiB":
                    subset[
                        "Max_Physical_VRAM"
                    ].max()
                    if
                    "Max_Physical_VRAM"
                    in subset.columns
                    else
                    np.nan,

                "Mean_GPU_Util_percent":
                    subset[
                        "Mean_GPU_Util"
                    ].mean()
                    if
                    "Mean_GPU_Util"
                    in subset.columns
                    else
                    np.nan,

                "Mean_Power_W":
                    subset[
                        "Mean_Power"
                    ].mean()
                    if
                    "Mean_Power"
                    in subset.columns
                    else
                    np.nan
            }
        )


summary_df = (
    pd.DataFrame(
        summary_rows
    )
)


# ============================================================
# PAIRED RESULTS
# ============================================================

paired_rows = []


for context_length in (
    PROMPT_LENGTHS
):

    context_df = results_df[
        results_df[
            "Context_Length"
        ]
        ==
        context_length
    ]


    pair_numbers = (
        sorted(
            context_df[
                "Pair"
            ].unique()
        )
    )


    for pair_number in (
        pair_numbers
    ):

        pair_df = context_df[
            context_df[
                "Pair"
            ]
            ==
            pair_number
        ]


        off_df = pair_df[
            pair_df[
                "KV_Cache"
            ]
            ==
            False
        ]


        on_df = pair_df[
            pair_df[
                "KV_Cache"
            ]
            ==
            True
        ]


        if (
            len(off_df) != 1
            or
            len(on_df) != 1
        ):

            raise RuntimeError(
                f"Incomplete pair "
                f"{pair_number}"
            )


        off_time = (
            off_df.iloc[0][
                "Time_s"
            ]
        )


        on_time = (
            on_df.iloc[0][
                "Time_s"
            ]
        )


        off_tps = (
            off_df.iloc[0][
                "Tokens_per_sec"
            ]
        )


        on_tps = (
            on_df.iloc[0][
                "Tokens_per_sec"
            ]
        )


        paired_rows.append(
            {

                "Context_Length":
                    context_length,

                "Pair":
                    pair_number,

                "Cache_OFF_Time_s":
                    off_time,

                "Cache_ON_Time_s":
                    on_time,

                "Time_Saved_s":
                    off_time
                    -
                    on_time,

                "Speedup":
                    off_time
                    /
                    on_time,

                "Cache_OFF_TPS":
                    off_tps,

                "Cache_ON_TPS":
                    on_tps,

                "TPS_Increase":
                    on_tps
                    -
                    off_tps
            }
        )


paired_df = (
    pd.DataFrame(
        paired_rows
    )
)


# ============================================================
# SPEEDUP SUMMARY
# ============================================================

print()

print("=" * 80)

print(
    "SPEEDUP SUMMARY"
)

print("=" * 80)


speedup_lines = []


for context_length in (
    PROMPT_LENGTHS
):

    subset = paired_df[
        paired_df[
            "Context_Length"
        ]
        ==
        context_length
    ]


    mean_speedup = (
        subset[
            "Speedup"
        ].mean()
    )


    median_speedup = (
        subset[
            "Speedup"
        ].median()
    )


    sd_speedup = (
        subset[
            "Speedup"
        ].std(
            ddof=1
        )
    )


    mean_time_saved = (
        subset[
            "Time_Saved_s"
        ].mean()
    )


    off_mean = (
        subset[
            "Cache_OFF_Time_s"
        ].mean()
    )


    on_mean = (
        subset[
            "Cache_ON_Time_s"
        ].mean()
    )


    runtime_reduction = (

        (
            off_mean
            -
            on_mean
        )
        /
        off_mean

        *
        100
    )


    line = (
        f"Context {context_length}: "
        f"mean paired speedup = "
        f"{mean_speedup:.6f}x"
        f" | median = "
        f"{median_speedup:.6f}x"
        f" | SD = "
        f"{sd_speedup:.6f}"
        f" | runtime reduction = "
        f"{runtime_reduction:.3f}%"
    )


    speedup_lines.append(
        line
    )


    print(
        line
    )


# ============================================================
# VRAM SAFETY CHECK
# ============================================================

if (
    "Max_Physical_VRAM"
    in results_df.columns
):

    max_physical_vram = (
        results_df[
            "Max_Physical_VRAM"
        ].max()
    )


    vram_percent = (

        max_physical_vram
        /
        physical_gpu_memory_mib

        *
        100
    )


    print()

    print("=" * 80)

    print(
        "VRAM SAFETY CHECK"
    )

    print("=" * 80)


    print(
        f"Maximum observed physical VRAM: "
        f"{max_physical_vram:.0f} MiB"
    )


    print(
        f"Physical capacity: "
        f"{physical_gpu_memory_mib:.0f} MiB"
    )


    print(
        f"Maximum utilization: "
        f"{vram_percent:.2f}%"
    )


    if vram_percent < 90:

        print(
            "STATUS: GOOD - fully resident"
        )

    elif vram_percent < 97:

        print(
            "STATUS: CAUTION"
        )

    else:

        print(
            "STATUS: WARNING - near capacity"
        )

else:

    max_physical_vram = np.nan
    vram_percent = np.nan


# ============================================================
# SAVE FINAL FILES
# ============================================================

results_df.to_csv(
    RAW_RESULTS_FILE,
    index=False
)


gpu_samples_df.to_csv(
    GPU_SAMPLES_FILE,
    index=False
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


paired_df.to_csv(
    PAIRED_FILE,
    index=False
)


# ============================================================
# ENVIRONMENT REPORT
# ============================================================

environment_lines = []


environment_lines.append(
    f"Timestamp: "
    f"{datetime.now().isoformat()}"
)


environment_lines.append(
    f"Model: "
    f"{MODEL_NAME}"
)


environment_lines.append(
    "Precision: FP16"
)


environment_lines.append(
    f"Python: "
    f"{sys.version}"
)


environment_lines.append(
    f"PyTorch: "
    f"{torch.__version__}"
)


environment_lines.append(
    f"Transformers: "
    f"{transformers.__version__}"
)


environment_lines.append(
    f"CUDA runtime: "
    f"{torch.version.cuda}"
)


environment_lines.append(
    f"GPU: "
    f"{torch.cuda.get_device_name(0)}"
)


environment_lines.append(
    f"Physical GPU memory: "
    f"{physical_gpu_memory_gib:.4f} GiB"
)


environment_lines.append(
    f"Model parameter memory: "
    f"{parameter_gib:.4f} GiB"
)


if hf_memory_gib is not None:

    environment_lines.append(
        f"HF model footprint: "
        f"{hf_memory_gib:.4f} GiB"
    )


environment_lines.append("")


environment_lines.append(
    "ARCHITECTURE:"
)


environment_lines.append(
    f"Layers: "
    f"{num_layers}"
)


environment_lines.append(
    f"Hidden size: "
    f"{hidden_size}"
)


environment_lines.append(
    f"Attention heads: "
    f"{num_attention_heads}"
)


environment_lines.append(
    f"KV heads: "
    f"{num_kv_heads}"
)


environment_lines.append(
    f"Head dimension: "
    f"{head_dimension}"
)


environment_lines.append("")


environment_lines.append(
    "KV CACHE:"
)


environment_lines.append(
    f"KV bytes/token: "
    f"{kv_bytes_per_token}"
)


environment_lines.append(
    f"KV KiB/token: "
    f"{kv_bytes_per_token / 1024:.6f}"
)


environment_lines.append("")


environment_lines.append(
    "EXPERIMENT:"
)


environment_lines.append(
    f"Prompt lengths: "
    f"{PROMPT_LENGTHS}"
)


environment_lines.append(
    f"Generated tokens/run: "
    f"{MAX_NEW_TOKENS}"
)


environment_lines.append(
    f"Runs per condition: "
    f"{RUNS_PER_CONDITION}"
)


environment_lines.append(
    f"Warmup runs: "
    f"{WARMUP_RUNS}"
)


environment_lines.append(
    f"Cooldown: "
    f"{COOLDOWN_SECONDS} seconds"
)


environment_lines.append(
    f"GPU sampling interval: "
    f"{GPU_SAMPLE_INTERVAL} seconds"
)


environment_lines.append("")


environment_lines.append(
    "NVIDIA-SMI BEFORE MODEL LOAD:"
)


environment_lines.append(
    str(
        initial_gpu_status
    )
)


environment_lines.append("")


environment_lines.append(
    "NVIDIA-SMI AFTER MODEL LOAD:"
)


environment_lines.append(
    str(
        gpu_after_model
    )
)


environment_lines.append("")


environment_lines.append(
    "SPEEDUP SUMMARY:"
)


environment_lines.extend(
    speedup_lines
)


if not np.isnan(
    vram_percent
):

    environment_lines.append("")

    environment_lines.append(
        "VRAM:"
    )

    environment_lines.append(
        f"Maximum physical VRAM: "
        f"{max_physical_vram:.0f} MiB"
    )

    environment_lines.append(
        f"Maximum utilization: "
        f"{vram_percent:.6f}%"
    )


with open(
    ENVIRONMENT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            environment_lines
        )
    )


# ============================================================
# FINISHED
# ============================================================

print()

print("=" * 80)

print(
    "SMOLLM3 CONTEXT SWEEP COMPLETE"
)

print("=" * 80)


print()

print(
    "Saved files:"
)


print(
    RAW_RESULTS_FILE
)

print(
    GPU_SAMPLES_FILE
)

print(
    SUMMARY_FILE
)

print(
    PAIRED_FILE
)

print(
    ENVIRONMENT_FILE
)
