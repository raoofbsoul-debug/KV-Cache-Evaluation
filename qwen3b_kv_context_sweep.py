import os

# ============================================================
# HUGGING FACE OFFLINE MODE
# ============================================================

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

MODEL_NAME = "Qwen/Qwen2.5-3B"

DEVICE = "cuda"

DTYPE = torch.float16

SEED = 42


# ------------------------------------------------------------
# CONTEXT LENGTHS TO TEST
# ------------------------------------------------------------

PROMPT_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
]


# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------

MAX_NEW_TOKENS = 128


# ------------------------------------------------------------
# EXPERIMENT REPETITIONS
# ------------------------------------------------------------

RUNS_PER_CONDITION = 3

WARMUP_RUNS = 2


# ------------------------------------------------------------
# GPU MONITORING
# ------------------------------------------------------------

GPU_SAMPLE_INTERVAL = 1.0


# ------------------------------------------------------------
# COOLDOWN
# ------------------------------------------------------------

COOLDOWN_SECONDS = 60


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_RESULTS_FILE = (
    "qwen3b_context_sweep_runs.csv"
)

GPU_SAMPLES_FILE = (
    "qwen3b_context_sweep_gpu_samples.csv"
)

SUMMARY_FILE = (
    "qwen3b_context_sweep_summary.csv"
)

PAIRED_FILE = (
    "qwen3b_context_sweep_paired.csv"
)

ENVIRONMENT_FILE = (
    "qwen3b_context_sweep_environment.txt"
)


# ============================================================
# BASE TEXT USED TO BUILD PROMPTS
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

    raise RuntimeError(
        "CUDA is not available."
    )


# ============================================================
# HELPERS
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

    print(
        "Cooldown complete."
    )


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

        "--query-gpu="
        +
        ",".join(fields),

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


        return dict(
            zip(
                fields,
                values
            )
        )


    except Exception as error:

        return {

            "error":
                str(error)

        }


# ============================================================
# PYTORCH MEMORY
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
        context_length,
        run_number,
        pair_number,
        use_cache,
        interval=1.0
    ):

        self.context_length = (
            context_length
        )

        self.run_number = (
            run_number
        )

        self.pair_number = (
            pair_number
        )

        self.use_cache = (
            use_cache
        )

        self.interval = interval

        self.samples = []

        self.stop_event = (
            threading.Event()
        )

        self.thread = None


    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    def sample_once(
        self,
        stage
    ):

        gpu = (
            get_nvidia_smi_stats()
        )


        torch_stats = (
            get_torch_memory_stats()
        )


        sample = {

            "Context_Length":
                self.context_length,

            "Run":
                self.run_number,

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


        sample.update(
            torch_stats
        )


        self.samples.append(
            sample
        )


    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    def _monitor_loop(self):

        while not self.stop_event.is_set():

            self.sample_once(
                stage="during"
            )

            self.stop_event.wait(
                self.interval
            )


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):

        self.sample_once(
            stage="before"
        )


        self.thread = (
            threading.Thread(

                target=
                    self._monitor_loop,

                daemon=True

            )
        )


        self.thread.start()


    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

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

print("=" * 78)

print(
    "QWEN2.5-3B FP16 KV-CACHE CONTEXT-LENGTH SWEEP"
)

print("=" * 78)


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

    "Context lengths:",
    PROMPT_LENGTHS

)


print(

    "Generated tokens:",
    MAX_NEW_TOKENS

)


print(

    "Runs per condition:",
    RUNS_PER_CONDITION

)


# ============================================================
# INITIAL NVIDIA-SMI
# ============================================================

print()

print("=" * 78)

print(
    "NVIDIA-SMI BEFORE MODEL LOAD"
)

print("=" * 78)


initial_gpu_status = (
    get_nvidia_smi_stats()
)


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

print("=" * 78)

print(
    "LOADING TOKENIZER"
)

print("=" * 78)


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

print("=" * 78)

print(
    "LOADING QWEN2.5-3B FP16"
)

print("=" * 78)


torch.cuda.empty_cache()


model = (
    AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        torch_dtype=
            DTYPE,

        device_map=
            "cuda",

        local_files_only=
            True

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

print("=" * 78)

print(
    "MODEL ARCHITECTURE"
)

print("=" * 78)


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


buffer_bytes = sum(

    buffer.numel()
    *
    buffer.element_size()

    for buffer
    in model.buffers()

)


parameter_gib = (
    bytes_to_gib(
        parameter_bytes
    )
)


print()

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
# THEORETICAL KV CACHE
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

print("=" * 78)

print(
    "THEORETICAL KV CACHE"
)

print("=" * 78)


print(

    f"KV bytes/token: "
    f"{kv_bytes_per_token:,}"

)


print(

    f"KV KiB/token: "
    f"{kv_bytes_per_token / 1024:.3f}"

)


print()

print(
    "Expected prompt KV cache:"
)


for context_length in PROMPT_LENGTHS:

    kv_mib = (

        context_length
        *
        kv_bytes_per_token
        /
        (1024 ** 2)

    )

    print(

        f"{context_length:5d} tokens: "
        f"{kv_mib:8.3f} MiB"

    )


# ============================================================
# BUILD EXACT-LENGTH PROMPT
# ============================================================

def create_exact_token_prompt(
    target_length
):

    """
    Create an input sequence containing exactly target_length
    tokenizer tokens.

    The same semantic base text is repeated until enough tokens
    exist, then the token sequence is truncated exactly.

    No special tokens are added, so target_length is precisely
    input_ids.shape[1].
    """


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
        token_ids[:target_length]
    )


    input_ids = torch.tensor(

        [token_ids],

        dtype=torch.long,

        device=DEVICE

    )


    attention_mask = torch.ones_like(
        input_ids
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
# VERIFY PROMPTS
# ============================================================

print()

print("=" * 78)

print(
    "PROMPT VERIFICATION"
)

print("=" * 78)


for context_length in PROMPT_LENGTHS:

    test_ids, test_mask = (
        create_exact_token_prompt(
            context_length
        )
    )


    print(

        f"Requested: "
        f"{context_length:5d} | "

        f"Actual: "
        f"{test_ids.shape[1]:5d}"

    )


    del test_ids
    del test_mask


torch.cuda.empty_cache()


# ============================================================
# BENCHMARK ONE RUN
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


    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()


    baseline_allocated = (
        torch.cuda.memory_allocated()
    )


    baseline_reserved = (
        torch.cuda.memory_reserved()
    )


    # --------------------------------------------------------
    # GPU MONITOR
    # --------------------------------------------------------

    monitor = GPUMonitor(

        context_length=
            context_length,

        run_number=
            run_number,

        pair_number=
            pair_number,

        use_cache=
            use_cache,

        interval=
            GPU_SAMPLE_INTERVAL

    )


    monitor.start()


    time.sleep(0.2)


    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    torch.cuda.synchronize()


    start_time = (
        time.perf_counter()
    )


    with torch.inference_mode():

        output_ids = model.generate(

            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            max_new_tokens=
                MAX_NEW_TOKENS,

            min_new_tokens=
                MAX_NEW_TOKENS,

            do_sample=
                False,

            use_cache=
                use_cache,

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


    result = {

        "Context_Length":
            context_length,

        "Run":
            run_number,

        "Pair":
            pair_number,

        "KV_Cache":
            use_cache,

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

            (
                context_length
                *
                kv_bytes_per_token
                /
                (1024 ** 2)
            ),

        "Theoretical_Final_KV_MiB":

            (
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

    }


    # --------------------------------------------------------
    # TELEMETRY SUMMARY
    # --------------------------------------------------------

    sample_df = pd.DataFrame(
        monitor.samples
    )


    during = sample_df[

        sample_df["Stage"]
        ==
        "during"

    ]


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

print("=" * 78)

print(
    "WARM-UP"
)

print("=" * 78)


# Use 256 tokens for warmup.

WARMUP_CONTEXT_LENGTH = 256


for warmup_index in range(
    1,
    WARMUP_RUNS + 1
):

    warmup_cache = bool(
        warmup_index % 2
    )


    input_ids, attention_mask = (
        create_exact_token_prompt(
            WARMUP_CONTEXT_LENGTH
        )
    )


    print()

    print(

        f"Warm-up "
        f"{warmup_index}/{WARMUP_RUNS}"

        f" | context="
        f"{WARMUP_CONTEXT_LENGTH}"

        f" | cache="
        f"{warmup_cache}"

    )


    torch.cuda.synchronize()


    with torch.inference_mode():

        warmup_output = model.generate(

            input_ids=
                input_ids,

            attention_mask=
                attention_mask,

            max_new_tokens=
                MAX_NEW_TOKENS,

            min_new_tokens=
                MAX_NEW_TOKENS,

            do_sample=
                False,

            use_cache=
                warmup_cache,

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
# BUILD RANDOMIZED SCHEDULE
# ============================================================

rng = random.Random(
    SEED
)


schedule = []


for context_length in PROMPT_LENGTHS:

    for pair_number in range(

        1,
        RUNS_PER_CONDITION + 1

    ):

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
                        pair_number,

                    "KV_Cache":
                        use_cache

                }
            )


# ============================================================
# DISPLAY SCHEDULE
# ============================================================

print()

print("=" * 78)

print(
    "EXPERIMENT SCHEDULE"
)

print("=" * 78)


print(

    f"Total measured runs: "
    f"{len(schedule)}"

)


for i, experiment in enumerate(
    schedule,
    start=1
):

    print(

        f"{i:02d}: "

        f"context="
        f"{experiment['Context_Length']:4d}"

        f" | pair="
        f"{experiment['Pair']}"

        f" | cache="
        f"{experiment['KV_Cache']}"

    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

all_gpu_samples = []


print()

print("=" * 78)

print(
    "STARTING CONTEXT-LENGTH SWEEP"
)

print("=" * 78)


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
        experiment["Pair"]
    )


    use_cache = (
        experiment["KV_Cache"]
    )


    print()

    print("-" * 78)


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

        f"Pair: "
        f"{pair_number}"

    )


    print(

        f"KV cache: "
        f"{use_cache}"

    )


    print("-" * 78)


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

            f"Physical VRAM max: "
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
        "Mean_GPU_Temp"
        in result
    ):

        print(

            f"Mean GPU temperature: "
            f"{result['Mean_GPU_Temp']:.1f} C"

        )


    # --------------------------------------------------------
    # SAVE AFTER EVERY RUN
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
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    results
)


gpu_samples_df = pd.DataFrame(
    all_gpu_samples
)


# ============================================================
# CONDITION SUMMARY
# ============================================================

summary_rows = []


print()

print("=" * 78)

print(
    "CONTEXT-LENGTH SUMMARY"
)

print("=" * 78)


for context_length in PROMPT_LENGTHS:

    print()

    print(
        "-" * 78
    )

    print(

        f"CONTEXT LENGTH = "
        f"{context_length}"

    )

    print(
        "-" * 78
    )


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


        mean_time = (
            subset["Time_s"].mean()
        )


        median_time = (
            subset["Time_s"].median()
        )


        sd_time = (
            subset["Time_s"].std(
                ddof=1
            )
        )


        mean_tps = (
            subset[
                "Tokens_per_sec"
            ].mean()
        )


        mean_peak = (
            subset[
                "Peak_Allocated_GiB"
            ].mean()
        )


        mean_reserved = (
            subset[
                "Peak_Reserved_GiB"
            ].mean()
        )


        row = {

            "Context_Length":
                context_length,

            "KV_Cache":
                use_cache,

            "N":
                len(subset),

            "Mean_Time_s":
                mean_time,

            "Median_Time_s":
                median_time,

            "SD_Time_s":
                sd_time,

            "Mean_Tokens_per_sec":
                mean_tps,

            "Mean_Peak_Allocated_GiB":
                mean_peak,

            "Mean_Peak_Reserved_GiB":
                mean_reserved

        }


        if (
            "Mean_Physical_VRAM"
            in subset.columns
        ):

            row[
                "Mean_Physical_VRAM_MiB"
            ] = (

                subset[
                    "Mean_Physical_VRAM"
                ].mean()

            )


        summary_rows.append(
            row
        )


        print()

        print(

            f"KV cache = "
            f"{use_cache}"

        )


        print(

            f"Mean runtime: "
            f"{mean_time:.4f} s"

        )


        print(

            f"Mean throughput: "
            f"{mean_tps:.3f} tok/s"

        )


        print(

            f"Mean peak allocated: "
            f"{mean_peak:.4f} GiB"

        )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# PAIRED SPEEDUP ANALYSIS
# ============================================================

paired_rows = []


print()

print("=" * 78)

print(
    "KV-CACHE SPEEDUP BY CONTEXT LENGTH"
)

print("=" * 78)


for context_length in PROMPT_LENGTHS:

    context_df = results_df[

        results_df[
            "Context_Length"
        ]
        ==
        context_length

    ]


    off_df = context_df[

        context_df[
            "KV_Cache"
        ]
        ==
        False

    ]


    on_df = context_df[

        context_df[
            "KV_Cache"
        ]
        ==
        True

    ]


    mean_off = (
        off_df[
            "Time_s"
        ].mean()
    )


    mean_on = (
        on_df[
            "Time_s"
        ].mean()
    )


    ratio_of_means = (

        mean_off
        /
        mean_on

    )


    runtime_reduction = (

        1
        -
        (
            mean_on
            /
            mean_off
        )

    ) * 100


    print()

    print(

        f"Context {context_length:4d}: "

        f"OFF={mean_off:.4f}s | "

        f"ON={mean_on:.4f}s | "

        f"Speedup="
        f"{ratio_of_means:.4f}x | "

        f"Reduction="
        f"{runtime_reduction:.2f}%"

    )


    for pair_number in range(

        1,
        RUNS_PER_CONDITION + 1

    ):

        pair_df = context_df[

            context_df["Pair"]
            ==
            pair_number

        ]


        off_pair = pair_df[

            pair_df[
                "KV_Cache"
            ]
            ==
            False

        ]


        on_pair = pair_df[

            pair_df[
                "KV_Cache"
            ]
            ==
            True

        ]


        if (
            len(off_pair) == 1
            and
            len(on_pair) == 1
        ):

            off_time = (
                off_pair.iloc[0][
                    "Time_s"
                ]
            )


            on_time = (
                on_pair.iloc[0][
                    "Time_s"
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
                        on_time

                }
            )


paired_df = pd.DataFrame(
    paired_rows
)


# ============================================================
# AGGREGATED SPEEDUP TABLE
# ============================================================

speedup_rows = []


for context_length in PROMPT_LENGTHS:

    subset = paired_df[

        paired_df[
            "Context_Length"
        ]
        ==
        context_length

    ]


    if len(subset) > 0:

        speedup_rows.append(
            {

                "Context_Length":
                    context_length,

                "Mean_Paired_Speedup":

                    subset[
                        "Speedup"
                    ].mean(),

                "Median_Paired_Speedup":

                    subset[
                        "Speedup"
                    ].median(),

                "Mean_Time_Saved_s":

                    subset[
                        "Time_Saved_s"
                    ].mean()

            }
        )


speedup_df = pd.DataFrame(
    speedup_rows
)


print()

print(
    speedup_df.to_string(
        index=False
    )
)


# ============================================================
# VRAM SAFETY
# ============================================================

if (
    "Max_Physical_VRAM"
    in results_df.columns
):

    maximum_vram = (

        results_df[
            "Max_Physical_VRAM"
        ].max()

    )


    vram_percentage = (

        maximum_vram
        /
        physical_gpu_memory_mib

        *
        100

    )


    print()

    print("=" * 78)

    print(
        "VRAM SAFETY CHECK"
    )

    print("=" * 78)


    print(

        f"Maximum physical VRAM: "
        f"{maximum_vram:.0f} MiB"

    )


    print(

        f"GPU capacity: "
        f"{physical_gpu_memory_mib:.0f} MiB"

    )


    print(

        f"Maximum utilization: "
        f"{vram_percentage:.2f}%"

    )


    if vram_percentage < 90:

        print(
            "STATUS: GOOD"
        )

    elif vram_percentage < 97:

        print(
            "STATUS: CAUTION"
        )

    else:

        print(
            "STATUS: WARNING"
        )


# ============================================================
# SAVE OUTPUTS
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

    "Prompt lengths: "
    +
    str(
        PROMPT_LENGTHS
    )

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


for _, row in (
    speedup_df.iterrows()
):

    environment_lines.append(

        f"Context "
        f"{int(row['Context_Length'])}: "

        f"mean paired speedup = "
        f"{row['Mean_Paired_Speedup']:.6f}x"

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

print("=" * 78)

print(
    "EXPERIMENT COMPLETE"
)

print("=" * 78)


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
