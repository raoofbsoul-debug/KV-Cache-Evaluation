# KV-Cache Performance Characterization for LLM Inference

This repository contains the experimental code and environment records used to characterize the performance and memory behavior of **key-value (KV) caching during autoregressive large language model inference**.

The experiments compare inference with the KV cache **enabled** and **disabled** while controlling prompt length, generation length, precision, GPU platform, warm-up runs, cooldown periods, and run order. The current repository includes experiments for:

- **Qwen2.5-3B** (`Qwen/Qwen2.5-3B`)
- **SmolLM3-3B** (`HuggingFaceTB/SmolLM3-3B`)

Both models are evaluated in **FP16** on an **NVIDIA GeForce RTX 4080 Laptop GPU** with approximately 12 GiB of physical GPU memory.

## Research Objective

The experiments investigate how KV caching affects autoregressive LLM inference and, in particular:

1. how cache-enabled and cache-disabled inference differ in runtime and throughput;
2. how the KV-cache performance benefit changes with prompt/context length;
3. how different grouped-query attention configurations affect KV-cache memory requirements;
4. whether the evaluated workloads remain resident in physical GPU memory; and
5. how GPU telemetry changes between cache-enabled and cache-disabled execution.

The central benchmarking principle is that KV-cache speedup should be interpreted together with **physical GPU-memory residency**, rather than from execution time alone.

## Models

| Model | Layers | Hidden Size | Query Heads | KV Heads | Head Dimension | FP16 KV Cache |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 36 | 2048 | 16 | 2 | 128 | 36 KiB/token |
| SmolLM3-3B | 36 | 2048 | 16 | 4 | 128 | 72 KiB/token |

For a Transformer with `L` layers, batch size `B`, sequence length `T`, `N_KV` key-value heads, head dimension `d_h`, and `s` bytes per KV element, the theoretical KV-cache storage is

```text
M_KV = 2 × L × B × T × N_KV × d_h × s
```

The factor of 2 accounts for both key and value tensors.

## Experiments

### 1. Context-length sweep

The context-sweep programs evaluate prompt lengths:

```text
128, 256, 512, 1024, 2048 tokens
```

with **128 generated tokens per run**. Each context condition uses warm-up runs followed by paired cache-enabled/cache-disabled measurements.

Scripts:

```text
qwen3b_kv_context_sweep.py
smollm3_3b_context_sweep.py
```

### 2. 2048-token validation experiment

The validation experiments use:

```text
Prompt length:       2048 tokens
Generated tokens:     128 tokens
Paired trials:         30
Measured runs:         60 per model
Warm-up runs:           3
GPU sampling interval:  1 second
Cooldown:              60 seconds
```

Scripts:

```text
qwen3b_fp16_2048_30runs.py
smollm3_3b_fp16_2048_30runs.py
```

The programs also perform statistical analyses of the paired measurements.

## GPU Telemetry

The benchmark programs collect GPU information using `nvidia-smi` and PyTorch CUDA memory statistics. Depending on the experiment, the collected telemetry includes:

- GPU utilization
- memory utilization
- physical GPU memory used
- physical GPU memory capacity
- GPU temperature
- SM clock
- memory clock
- power draw
- performance state
- PyTorch allocated memory
- PyTorch reserved memory
- peak PyTorch allocated/reserved memory

This telemetry is used to verify GPU-memory residency and to help distinguish computational KV-cache effects from memory-capacity effects.

## Selected Results

### Context-length sweep

| Context | Qwen2.5-3B Mean Paired Speedup | SmolLM3-3B Mean Paired Speedup |
|---:|---:|---:|
| 128 | 1.337× | 1.278× |
| 256 | 1.687× | 1.728× |
| 512 | 2.607× | 2.648× |
| 1024 | 4.343× | 4.653× |
| 2048 | 8.314× | 8.619× |

The sweep shows a strong increase in the measured benefit of KV caching as context length grows.

### 2048-token, 30-pair validation

| Metric | Qwen2.5-3B | SmolLM3-3B |
|---|---:|---:|
| Cache OFF mean runtime | 30.471 s | 30.703 s |
| Cache ON mean runtime | 3.762 s | 3.609 s |
| Mean paired speedup | 8.101× | 8.511× |
| Runtime reduction | 87.654% | 88.247% |
| Maximum physical VRAM | 7434 MiB | 7459 MiB |
| Maximum VRAM utilization | 60.53% | 60.73% |

These measurements indicate that the 2048-token validation workloads remained well below the physical GPU-memory capacity of the test device.

## Repository Files

```text
.
├── qwen3b_kv_context_sweep.py
├── qwen3b_fp16_2048_30runs.py
├── smollm3_3b_context_sweep.py
├── smollm3_3b_fp16_2048_30runs.py
├── qwen3b_context_sweep_environment.txt
├── qwen3b_fp16_2048_30runs_environment.txt
├── qwen3b_fp16_diagnostic_environment.txt
├── smollm3_3b_context_sweep_environment.txt
├── smollm3_3b_fp16_2048_30runs_environment.txt
└── README.md
```

The `*_environment.txt` files record the software environment, model architecture, experimental configuration, KV-cache calculations, GPU state, and selected performance summaries from the corresponding runs.

## Software Environment

The recorded experiments used:

```text
Python:       3.14.4
PyTorch:      2.14.0+cu132
Transformers: 5.16.1
CUDA runtime: 13.2
GPU:          NVIDIA GeForce RTX 4080 Laptop GPU
```

The Python scripts additionally use packages including:

```text
numpy
pandas
torch
transformers
tqdm
scipy
```

`nvidia-smi` must be available for NVIDIA GPU telemetry.

## Running the Experiments

The model files must be available locally because the scripts enable Hugging Face offline mode and load models using `local_files_only=True`.

For example:

```bash
python qwen3b_kv_context_sweep.py
```

or:

```bash
python qwen3b_fp16_2048_30runs.py
```

For SmolLM3-3B:

```bash
python smollm3_3b_context_sweep.py
python smollm3_3b_fp16_2048_30runs.py
```

The scripts generate CSV files containing raw measurements, paired results, summaries, and GPU telemetry, together with environment records.

## Reproducibility Notes

The benchmark scripts use a fixed random seed (`42`) and explicitly synchronize CUDA around timed operations. They include warm-up runs and cooldown intervals to reduce initialization and thermal effects. Cache-enabled/cache-disabled conditions are paired, and the validation experiments use randomized paired execution.

Exact performance may differ across GPUs, driver versions, operating systems, model/library versions, power limits, and thermal conditions. The environment files are therefore retained as part of the experimental record.

## Interpretation

The results should not be interpreted as a universal constant speedup for KV caching. KV-cache benefit depends on factors including:

- context length;
- generation length;
- model architecture;
- number of KV heads;
- numerical precision;
- inference implementation;
- accelerator characteristics; and
- memory residency.

In particular, experiments conducted near or beyond physical GPU-memory capacity may include paging, migration, offloading, or other memory-system effects. Such effects can confound an attempt to attribute the entire observed runtime difference solely to KV reuse.

## Citation

If you use this repository in academic work, please cite the associated paper once its bibliographic information is available.

```bibtex
@article{kv_cache_characterization,
  title   = {Characterizing the Memory--Performance Tradeoff of KV Caching in GPU-Based Large Language Model Inference},
  author  = {To be added},
  journal = {To be added},
  year    = {To be added}
}
```

## License

No license has been specified in this repository yet. Add an appropriate `LICENSE` file before public release if you intend to permit reuse, modification, or redistribution of the code.
