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
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B"

DEVICE = "cuda"
DTYPE = torch.float16

SEED = 42

PROMPT_LENGTH = 2048
MAX_NEW_TOKENS = 128

NUM_PAIRS = 30

WARMUP_RUNS = 3

COOLDOWN_SECONDS = 60
GPU_SAMPLE_INTERVAL = 1.0


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_RESULTS_FILE = "qwen3b_fp16_2048_30runs_raw.csv"

GPU_SAMPLES_FILE = "qwen3b_fp16_2048_30runs_gpu_samples.csv"

PAIRED_FILE = "qwen3b_fp16_2048_30runs_paired.csv"

SUMMARY_FILE = "qwen3b_fp16_2048_30runs_summary.csv"

ENVIRONMENT_FILE = "qwen3b_fp16_2048_30runs_environment.txt"


# ============================================================
# BASE TEXT
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
        run_number,
        pair_number,
        use_cache,
        interval=1.0
    ):

        self.run_number = run_number
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

            "Run": self.run_number,

            "Pair": self.pair_number,

            "KV_Cache": self.use_cache,

            "Stage": stage,

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
    "QWEN2.5-3B FP16 KV-CACHE 30+30 BENCHMARK"
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
    f"Prompt tokens: "
    f"{PROMPT_LENGTH}"
)

print(
    f"Generated tokens/run: "
    f"{MAX_NEW_TOKENS}"
)

print(
    f"Pairs: "
    f"{NUM_PAIRS}"
)

print(
    f"Total measured runs: "
    f"{NUM_PAIRS * 2}"
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
        torch_dtype=DTYPE,
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


prompt_kv_mib = (

    PROMPT_LENGTH
    *
    kv_bytes_per_token
    /
    (1024 ** 2)
)


final_kv_mib = (

    (
        PROMPT_LENGTH
        +
        MAX_NEW_TOKENS
    )
    *
    kv_bytes_per_token
    /
    (1024 ** 2)
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

print(
    f"Prompt KV: "
    f"{prompt_kv_mib:.3f} MiB"
)

print(
    f"Approx final KV: "
    f"{final_kv_mib:.3f} MiB"
)


# ============================================================
# BUILD EXACT 2048-TOKEN INPUT
# ============================================================

def create_exact_token_prompt():

    base_ids = tokenizer.encode(
        BASE_TEXT,
        add_special_tokens=False
    )


    if len(base_ids) == 0:

        raise RuntimeError(
            "Base text produced no tokens."
        )


    repeats = (
        PROMPT_LENGTH
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
        token_ids[:PROMPT_LENGTH]
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
        PROMPT_LENGTH
    )


    return (
        input_ids,
        attention_mask
    )


# ============================================================
# VERIFY INPUT
# ============================================================

verification_ids, verification_mask = (
    create_exact_token_prompt()
)


print()

print(
    f"Requested prompt tokens: "
    f"{PROMPT_LENGTH}"
)

print(
    f"Actual prompt tokens: "
    f"{verification_ids.shape[1]}"
)


del verification_ids
del verification_mask

torch.cuda.empty_cache()


# ============================================================
# ONE BENCHMARK RUN
# ============================================================

def benchmark_once(
    use_cache,
    run_number,
    pair_number
):

    input_ids, attention_mask = (
        create_exact_token_prompt()
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
        PROMPT_LENGTH
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

        "Run":
            run_number,

        "Pair":
            pair_number,

        "KV_Cache":
            use_cache,

        "Prompt_Tokens":
            PROMPT_LENGTH,

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
            prompt_kv_mib,

        "Theoretical_Final_KV_MiB":
            final_kv_mib
    }


    sample_df = (
        pd.DataFrame(
            monitor.samples
        )
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
# WARMUP
# ============================================================

print()

print("=" * 80)

print(
    "WARM-UP"
)

print("=" * 80)


warmup_conditions = [
    True,
    False,
    True
]


for warmup_index in range(
    WARMUP_RUNS
):

    use_cache = (
        warmup_conditions[
            warmup_index
            %
            len(warmup_conditions)
        ]
    )


    input_ids, attention_mask = (
        create_exact_token_prompt()
    )


    print()

    print(
        f"Warm-up "
        f"{warmup_index + 1}/"
        f"{WARMUP_RUNS}"
        f" | cache={use_cache}"
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
# RANDOMIZED PAIRED SCHEDULE
# ============================================================

rng = random.Random(
    SEED
)


schedule = []


for pair_number in range(
    1,
    NUM_PAIRS + 1
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
                "Pair":
                    pair_number,

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


for i, experiment in enumerate(
    schedule,
    start=1
):

    print(
        f"{i:02d}: "
        f"Pair={experiment['Pair']:02d}"
        f" | Cache="
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
    "STARTING 30+30 BENCHMARK"
)

print("=" * 80)


for run_number, experiment in enumerate(
    schedule,
    start=1
):

    pair_number = (
        experiment["Pair"]
    )

    use_cache = (
        experiment[
            "KV_Cache"
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
            use_cache=use_cache,
            run_number=run_number,
            pair_number=pair_number
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
        "Mean_GPU_Temp"
        in result
    ):

        print(
            f"Mean GPU temperature: "
            f"{result['Mean_GPU_Temp']:.1f} C"
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
    # SAVE PARTIAL RESULTS
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
    # COOLDOWN BETWEEN RUNS
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


off_df = results_df[
    results_df["KV_Cache"]
    ==
    False
].copy()


on_df = results_df[
    results_df["KV_Cache"]
    ==
    True
].copy()


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize_condition(
    df,
    name
):

    times = (
        df["Time_s"]
        .to_numpy()
    )


    tps = (
        df["Tokens_per_sec"]
        .to_numpy()
    )


    n = len(times)


    mean_time = (
        np.mean(times)
    )


    median_time = (
        np.median(times)
    )


    sd_time = (
        np.std(
            times,
            ddof=1
        )
    )


    sem_time = (
        stats.sem(times)
    )


    ci_time = (
        stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=mean_time,
            scale=sem_time
        )
    )


    mean_tps = (
        np.mean(tps)
    )


    median_tps = (
        np.median(tps)
    )


    sd_tps = (
        np.std(
            tps,
            ddof=1
        )
    )


    return {

        "Condition":
            name,

        "N":
            n,

        "Mean_Time_s":
            mean_time,

        "Median_Time_s":
            median_time,

        "SD_Time_s":
            sd_time,

        "Min_Time_s":
            np.min(times),

        "Max_Time_s":
            np.max(times),

        "CI95_Time_Lower":
            ci_time[0],

        "CI95_Time_Upper":
            ci_time[1],

        "Mean_Tokens_per_sec":
            mean_tps,

        "Median_Tokens_per_sec":
            median_tps,

        "SD_Tokens_per_sec":
            sd_tps,

        "Min_Tokens_per_sec":
            np.min(tps),

        "Max_Tokens_per_sec":
            np.max(tps),

        "Mean_Peak_Allocated_GiB":
            df[
                "Peak_Allocated_GiB"
            ].mean(),

        "Mean_Peak_Reserved_GiB":
            df[
                "Peak_Reserved_GiB"
            ].mean(),

        "Mean_Incremental_Peak_MiB":
            df[
                "Incremental_Peak_Allocated_MiB"
            ].mean()
    }


summary_rows = [

    summarize_condition(
        off_df,
        "Cache OFF"
    ),

    summarize_condition(
        on_df,
        "Cache ON"
    )
]


summary_df = (
    pd.DataFrame(
        summary_rows
    )
)


# ============================================================
# PAIRED DATA
# ============================================================

paired_rows = []


for pair_number in range(
    1,
    NUM_PAIRS + 1
):

    pair_df = results_df[
        results_df["Pair"]
        ==
        pair_number
    ]


    off_pair = pair_df[
        pair_df["KV_Cache"]
        ==
        False
    ]


    on_pair = pair_df[
        pair_df["KV_Cache"]
        ==
        True
    ]


    if (
        len(off_pair) != 1
        or
        len(on_pair) != 1
    ):

        raise RuntimeError(
            f"Pair {pair_number} is incomplete."
        )


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


    off_tps = (
        off_pair.iloc[0][
            "Tokens_per_sec"
        ]
    )


    on_tps = (
        on_pair.iloc[0][
            "Tokens_per_sec"
        ]
    )


    paired_rows.append(
        {

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
# PAIRED STATISTICAL TESTS
# ============================================================

off_times = (
    paired_df[
        "Cache_OFF_Time_s"
    ].to_numpy()
)


on_times = (
    paired_df[
        "Cache_ON_Time_s"
    ].to_numpy()
)


differences = (
    off_times
    -
    on_times
)


# ------------------------------------------------------------
# Paired t-test
# ------------------------------------------------------------

paired_t = (
    stats.ttest_rel(
        off_times,
        on_times
    )
)


# ------------------------------------------------------------
# Wilcoxon signed-rank
# ------------------------------------------------------------

wilcoxon_test = (
    stats.wilcoxon(
        off_times,
        on_times,
        alternative="greater"
    )
)


# ------------------------------------------------------------
# Cohen's dz
# ------------------------------------------------------------

cohen_dz = (

    np.mean(differences)
    /
    np.std(
        differences,
        ddof=1
    )
)


# ============================================================
# DERIVED EFFECTS
# ============================================================

mean_off_time = (
    off_df["Time_s"]
    .mean()
)


mean_on_time = (
    on_df["Time_s"]
    .mean()
)


ratio_of_means = (
    mean_off_time
    /
    mean_on_time
)


runtime_reduction_percent = (

    (
        mean_off_time
        -
        mean_on_time
    )
    /
    mean_off_time

    *
    100
)


mean_off_tps = (
    off_df[
        "Tokens_per_sec"
    ].mean()
)


mean_on_tps = (
    on_df[
        "Tokens_per_sec"
    ].mean()
)


throughput_increase_percent = (

    (
        mean_on_tps
        -
        mean_off_tps
    )
    /
    mean_off_tps

    *
    100
)


mean_paired_speedup = (
    paired_df[
        "Speedup"
    ].mean()
)


median_paired_speedup = (
    paired_df[
        "Speedup"
    ].median()
)


sd_paired_speedup = (
    paired_df[
        "Speedup"
    ].std(
        ddof=1
    )
)


mean_time_saved = (
    paired_df[
        "Time_Saved_s"
    ].mean()
)


median_time_saved = (
    paired_df[
        "Time_Saved_s"
    ].median()
)


sd_time_saved = (
    paired_df[
        "Time_Saved_s"
    ].std(
        ddof=1
    )
)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def iqr_outliers(values):

    values = np.asarray(
        values
    )

    q1 = np.percentile(
        values,
        25
    )

    q3 = np.percentile(
        values,
        75
    )

    iqr = (
        q3
        -
        q1
    )

    lower = (
        q1
        -
        1.5
        *
        iqr
    )

    upper = (
        q3
        +
        1.5
        *
        iqr
    )


    mask = (
        (values < lower)
        |
        (values > upper)
    )


    return {

        "Q1":
            q1,

        "Q3":
            q3,

        "IQR":
            iqr,

        "Lower":
            lower,

        "Upper":
            upper,

        "Outliers":
            values[mask]
    }


off_outliers = (
    iqr_outliers(
        off_times
    )
)


on_outliers = (
    iqr_outliers(
        on_times
    )
)


# ============================================================
# TREND ANALYSIS
# ============================================================

off_sorted = (
    off_df.sort_values(
        "Run"
    )
)


on_sorted = (
    on_df.sort_values(
        "Run"
    )
)


off_spearman = (
    stats.spearmanr(
        off_sorted["Run"],
        off_sorted["Time_s"]
    )
)


on_spearman = (
    stats.spearmanr(
        on_sorted["Run"],
        on_sorted["Time_s"]
    )
)


# ============================================================
# VRAM CHECK
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

else:

    maximum_vram = np.nan
    vram_percentage = np.nan


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print("=" * 80)

print(
    "FINAL PERFORMANCE SUMMARY"
)

print("=" * 80)


print()

print(
    summary_df.to_string(
        index=False
    )
)


print()

print(
    f"Ratio of mean runtimes: "
    f"{ratio_of_means:.6f}x"
)


print(
    f"Mean paired speedup: "
    f"{mean_paired_speedup:.6f}x"
)


print(
    f"Median paired speedup: "
    f"{median_paired_speedup:.6f}x"
)


print(
    f"SD paired speedup: "
    f"{sd_paired_speedup:.6f}"
)


print(
    f"Runtime reduction: "
    f"{runtime_reduction_percent:.3f}%"
)


print(
    f"Throughput improvement: "
    f"{throughput_increase_percent:.3f}%"
)


print(
    f"Mean time saved: "
    f"{mean_time_saved:.6f} s"
)


print(
    f"Median time saved: "
    f"{median_time_saved:.6f} s"
)


print(
    f"SD time saved: "
    f"{sd_time_saved:.6f} s"
)


print()

print("=" * 80)

print(
    "STATISTICAL TESTS"
)

print("=" * 80)


print(
    f"Paired t-test:"
)

print(
    f"t = "
    f"{paired_t.statistic:.6f}"
)

print(
    f"p = "
    f"{paired_t.pvalue:.12e}"
)

print(
    f"df = "
    f"{NUM_PAIRS - 1}"
)


print()

print(
    "Wilcoxon signed-rank:"
)

print(
    f"statistic = "
    f"{wilcoxon_test.statistic:.6f}"
)

print(
    f"p = "
    f"{wilcoxon_test.pvalue:.12e}"
)


print()

print(
    f"Cohen dz = "
    f"{cohen_dz:.6f}"
)


print()

print("=" * 80)

print(
    "OUTLIER ANALYSIS"
)

print("=" * 80)


print()

print(
    "Cache OFF:"
)

print(
    f"Q1 = "
    f"{off_outliers['Q1']:.6f}"
)

print(
    f"Q3 = "
    f"{off_outliers['Q3']:.6f}"
)

print(
    f"IQR = "
    f"{off_outliers['IQR']:.6f}"
)

print(
    f"Upper fence = "
    f"{off_outliers['Upper']:.6f}"
)

print(
    f"Outliers = "
    f"{off_outliers['Outliers']}"
)


print()

print(
    "Cache ON:"
)

print(
    f"Q1 = "
    f"{on_outliers['Q1']:.6f}"
)

print(
    f"Q3 = "
    f"{on_outliers['Q3']:.6f}"
)

print(
    f"IQR = "
    f"{on_outliers['IQR']:.6f}"
)

print(
    f"Upper fence = "
    f"{on_outliers['Upper']:.6f}"
)

print(
    f"Outliers = "
    f"{on_outliers['Outliers']}"
)


print()

print("=" * 80)

print(
    "RUN-ORDER TREND"
)

print("=" * 80)


print(
    f"Cache OFF Spearman rho = "
    f"{off_spearman.statistic:.6f}"
)

print(
    f"Cache OFF p = "
    f"{off_spearman.pvalue:.6e}"
)


print(
    f"Cache ON Spearman rho = "
    f"{on_spearman.statistic:.6f}"
)

print(
    f"Cache ON p = "
    f"{on_spearman.pvalue:.6e}"
)


if not np.isnan(
    vram_percentage
):

    print()

    print("=" * 80)

    print(
        "VRAM SAFETY CHECK"
    )

    print("=" * 80)


    print(
        f"Maximum physical VRAM: "
        f"{maximum_vram:.0f} MiB"
    )


    print(
        f"Physical capacity: "
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
# SAVE FILES
# ============================================================

results_df.to_csv(
    RAW_RESULTS_FILE,
    index=False
)


gpu_samples_df.to_csv(
    GPU_SAMPLES_FILE,
    index=False
)


paired_df.to_csv(
    PAIRED_FILE,
    index=False
)


summary_df.to_csv(
    SUMMARY_FILE,
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
    "MODEL ARCHITECTURE:"
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
    "EXPERIMENT:"
)


environment_lines.append(
    f"Prompt tokens: "
    f"{PROMPT_LENGTH}"
)


environment_lines.append(
    f"Generated tokens/run: "
    f"{MAX_NEW_TOKENS}"
)


environment_lines.append(
    f"Pairs: "
    f"{NUM_PAIRS}"
)


environment_lines.append(
    f"Runs per condition: "
    f"{NUM_PAIRS}"
)


environment_lines.append(
    f"Total measured runs: "
    f"{NUM_PAIRS * 2}"
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


environment_lines.append(
    f"Prompt KV MiB: "
    f"{prompt_kv_mib:.6f}"
)


environment_lines.append(
    f"Final KV MiB: "
    f"{final_kv_mib:.6f}"
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
    "PERFORMANCE SUMMARY:"
)


environment_lines.append(
    f"Cache OFF mean runtime: "
    f"{mean_off_time:.6f} s"
)


environment_lines.append(
    f"Cache ON mean runtime: "
    f"{mean_on_time:.6f} s"
)


environment_lines.append(
    f"Ratio of mean runtimes: "
    f"{ratio_of_means:.6f}x"
)


environment_lines.append(
    f"Mean paired speedup: "
    f"{mean_paired_speedup:.6f}x"
)


environment_lines.append(
    f"Median paired speedup: "
    f"{median_paired_speedup:.6f}x"
)


environment_lines.append(
    f"Runtime reduction: "
    f"{runtime_reduction_percent:.6f}%"
)


environment_lines.append(
    f"Throughput improvement: "
    f"{throughput_increase_percent:.6f}%"
)


environment_lines.append(
    f"Mean time saved: "
    f"{mean_time_saved:.6f} s"
)


environment_lines.append("")


environment_lines.append(
    "STATISTICAL TESTS:"
)


environment_lines.append(
    f"Paired t statistic: "
    f"{paired_t.statistic:.6f}"
)


environment_lines.append(
    f"Paired t p-value: "
    f"{paired_t.pvalue:.12e}"
)


environment_lines.append(
    f"Wilcoxon statistic: "
    f"{wilcoxon_test.statistic:.6f}"
)


environment_lines.append(
    f"Wilcoxon p-value: "
    f"{wilcoxon_test.pvalue:.12e}"
)


environment_lines.append(
    f"Cohen dz: "
    f"{cohen_dz:.6f}"
)


environment_lines.append("")


environment_lines.append(
    "RUN-ORDER TREND:"
)


environment_lines.append(
    f"Cache OFF Spearman rho: "
    f"{off_spearman.statistic:.6f}"
)


environment_lines.append(
    f"Cache OFF Spearman p: "
    f"{off_spearman.pvalue:.12e}"
)


environment_lines.append(
    f"Cache ON Spearman rho: "
    f"{on_spearman.statistic:.6f}"
)


environment_lines.append(
    f"Cache ON Spearman p: "
    f"{on_spearman.pvalue:.12e}"
)


if not np.isnan(
    vram_percentage
):

    environment_lines.append("")

    environment_lines.append(
        "VRAM:"
    )

    environment_lines.append(
        f"Maximum physical VRAM: "
        f"{maximum_vram:.0f} MiB"
    )

    environment_lines.append(
        f"Maximum utilization: "
        f"{vram_percentage:.6f}%"
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
    "30+30 EXPERIMENT COMPLETE"
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
    PAIRED_FILE
)

print(
    SUMMARY_FILE
)

print(
    ENVIRONMENT_FILE
)
