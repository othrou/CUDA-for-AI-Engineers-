Absolutely. I’d make `resources/README.md` a **curated learning hub**, rather than simply a list of links. That makes it useful to students who want to go deeper into profiling, CUDA optimization, and distributed training.

````markdown
# 📚 Resources

A curated collection of documentation, tutorials, articles, and references related to CUDA, PyTorch performance optimization, GPU profiling, and distributed training.

The resources are organized by topic so you can use them to deepen your understanding, explore concepts beyond the projects, or find references when working on your own experiments.

> **Learn → Experiment → Profile → Optimize**

---

## 🔬 PyTorch Profiling

### PyTorch Profiler — Official Documentation

The official PyTorch recipe for using `torch.profiler` to measure execution time and memory consumption, analyze operators, inspect CUDA activity, and generate execution traces.

🔗 https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

**Recommended for:** understanding the official PyTorch Profiler API and its main capabilities.

---

### Profiling GPU-accelerated Deep Learning — Sigma2

A practical introduction to profiling GPU-accelerated deep learning applications with PyTorch Profiler.

The guide covers GPU usage, CUDA kernels, memory activity, trace visualization, and TensorBoard-based analysis.

🔗 https://documentation.sigma2.no/code_development/guides/pytorch_profiler.html

**Recommended for:** learning how to interpret profiling information and identify performance bottlenecks.

---

### Profiling in PyTorch — Hugging Face

A detailed, practical introduction to reading PyTorch profiler traces.

The article goes beyond simply running the profiler and explains how to interpret CPU lanes, GPU lanes, CUDA kernels, dispatch chains, synchronization, and profiling traces.

🔗 https://huggingface.co/blog/torch-profiler

**Recommended for:** developing the ability to actually read and understand profiler traces.

---

### PyTorch Profiler API

The official API reference for `torch.profiler.profile`.

Useful when you need to understand options such as:

- `activities`
- `schedule`
- `record_shapes`
- `profile_memory`
- `with_stack`
- `with_flops`
- `with_modules`

🔗 https://docs.pytorch.org/docs/stable/profiler.html

**Recommended for:** reference and advanced profiling experiments.

---

## 📊 Profiling & GPU Performance

### NVIDIA Nsight Systems / Nsight Compute

NVIDIA's profiling tools provide a lower-level view of GPU and CPU execution.

They can help investigate:

- CUDA kernels
- GPU utilization
- memory transfers
- CPU-GPU synchronization
- kernel launch overhead
- execution timelines
- GPU performance characteristics

🔗 https://developer.nvidia.com/nsight-ai

**Recommended for:** going beyond PyTorch-level profiling and understanding what is happening at the GPU level.

---

### PyTorch Profiling with TensorBoard

PyTorch Profiler can export traces that can be visualized through TensorBoard.

🔗 https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

**Recommended for:** interactive visualization of profiling results.

---

## ⚡ CUDA & PyTorch Performance

### PyTorch CUDA Documentation

Official documentation covering CUDA-specific functionality in PyTorch.

Topics include:

- CUDA devices
- CUDA tensors
- streams
- events
- memory management
- synchronization
- CUDA semantics

🔗 https://docs.pytorch.org/docs/stable/notes/cuda.html

**Recommended for:** understanding how PyTorch interacts with CUDA.

---

### PyTorch CUDA API

Reference documentation for the `torch.cuda` package.

🔗 https://docs.pytorch.org/docs/stable/cuda.html

Useful functions include:

```python
torch.cuda.is_available()
torch.cuda.get_device_name()
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.synchronize()
```
````

**Recommended for:** GPU inspection, memory monitoring, and CUDA programming with PyTorch.

---

## 🧠 Mixed Precision & AMP

### Automatic Mixed Precision — PyTorch

Official documentation for Automatic Mixed Precision in PyTorch.

AMP allows compatible operations to use lower precision while maintaining numerical stability where required.

🔗 [https://docs.pytorch.org/docs/stable/amp.html](https://docs.pytorch.org/docs/stable/amp.html)

**Recommended for:** understanding FP16, BF16, autocast, and gradient scaling.

---

### Automatic Mixed Precision Recipe

A practical PyTorch recipe demonstrating how to use AMP during training.

🔗 [https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

**Recommended for:** implementing AMP in real training pipelines.

---

## 🚚 Data Loading & Memory Transfers

### PyTorch DataLoader

Official documentation for PyTorch's data loading utilities.

Topics include:

- `num_workers`
- `pin_memory`
- `prefetch_factor`
- batching
- multiprocessing

🔗 [https://docs.pytorch.org/docs/stable/data.html](https://docs.pytorch.org/docs/stable/data.html)

**Recommended for:** optimizing input pipelines and reducing CPU-side bottlenecks.

---

### PyTorch Memory Management

Documentation explaining PyTorch's CUDA memory allocator and memory management behavior.

🔗 [https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)

Useful when investigating:

- allocated memory
- reserved memory
- caching
- memory fragmentation
- `empty_cache()`

---

## 🌐 Distributed & Multi-GPU Training

### PyTorch Distributed Data Parallel

Official PyTorch introduction to DistributedDataParallel.

It explains how models are replicated across GPUs, how each device processes different batches, and how gradients are synchronized.

🔗 [https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html)

**Recommended for:** understanding the fundamentals of multi-GPU data parallelism.

---

### Training a Model on Multiple GPUs with Data Parallelism

A practical introduction to training models across multiple GPUs using data parallelism.

🔗 [https://machinelearningmastery.com/training-a-model-on-multiple-gpus-with-data-parallelism/](https://machinelearningmastery.com/training-a-model-on-multiple-gpus-with-data-parallelism/)

**Recommended for:** getting an additional practical perspective on multi-GPU training.

---

### Hugging Face — Training on Multiple GPUs

Hugging Face documentation covering multi-GPU training and distributed training strategies.

🔗 [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)

**Recommended for:** understanding how multi-GPU training is applied to modern Transformer workloads.

---

### PyTorch Distributed Overview

Official PyTorch documentation covering distributed communication and distributed training APIs.

🔗 [https://docs.pytorch.org/docs/stable/distributed.html](https://docs.pytorch.org/docs/stable/distributed.html)

**Recommended for:** deeper exploration of PyTorch distributed training.

---

## 🧩 CUDA Programming

### CUDA C++ Programming Guide

The official NVIDIA guide to CUDA programming.

It covers fundamental concepts such as:

- kernels
- threads
- blocks
- grids
- memory hierarchy
- synchronization
- execution model

🔗 [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

**Recommended for:** students who want to go deeper into CUDA beyond PyTorch.

---

### CUDA Toolkit Documentation

Official NVIDIA documentation for the CUDA Toolkit.

🔗 [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)

**Recommended for:** exploring CUDA libraries, tools, programming interfaces, and developer utilities.

---

## 🔥 PyTorch Optimization

### PyTorch Performance Tuning Guide

A collection of recommendations for improving PyTorch performance.

Topics include:

- DataLoader optimization
- pinned memory
- asynchronous transfers
- distributed training
- CUDA benchmarking
- memory optimization

🔗 [https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

**Recommended for:** building a broader performance optimization mindset.

---

### PyTorch `torch.compile`

Official documentation for PyTorch compilation and graph optimization.

🔗 [https://docs.pytorch.org/docs/stable/torch.compiler.html](https://docs.pytorch.org/docs/stable/torch.compiler.html)

**Recommended for:** exploring optimization techniques beyond the approaches covered in the initial projects.

---

## 📈 Benchmarking

### PyTorch Benchmark Utilities

PyTorch provides benchmarking utilities designed for more reliable performance measurements.

🔗 [https://docs.pytorch.org/docs/stable/benchmark_utils.html](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

**Recommended for:** writing more rigorous benchmarks and reducing measurement errors.

---

### CUDA Events for GPU Timing

CUDA events can be used to measure GPU execution time more accurately than relying only on CPU timers.

🔗 [https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)

**Recommended for:** accurate GPU timing and CUDA performance experiments.

---

## 🤗 Hugging Face Performance Resources

### Hugging Face Performance Documentation

Hugging Face maintains several resources covering GPU optimization, distributed training, mixed precision, and efficient model execution.

🔗 [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)

🔗 [https://huggingface.co/blog/torch-profiler](https://huggingface.co/blog/torch-profiler)

**Recommended for:** connecting the concepts learned here with modern Transformer and LLM workloads.

---

## 🧭 Suggested Learning Path

If you're not sure where to start, follow this order:

### 1. CUDA Fundamentals

Start with the PyTorch CUDA documentation and understand:

```text
CPU
 ↓
GPU
 ↓
CUDA Device
 ↓
CUDA Tensor
 ↓
CUDA Kernel
```

---

### 2. GPU Performance

Learn how to measure:

```text
Execution Time
GPU Utilization
Memory Usage
CPU-GPU Transfers
```

---

### 3. Profiling

Start with:

1. PyTorch Profiler
2. TensorBoard
3. Profiler traces
4. CUDA kernels
5. NVIDIA Nsight

---

### 4. Optimization

Experiment with:

```text
GPU
 ↓
Pinned Memory
 ↓
Async Transfers
 ↓
Batch Size
 ↓
DataLoader Workers
 ↓
AMP
 ↓
torch.compile
```

---

### 5. Distributed Training

Once single-GPU optimization is understood, move toward:

```text
Single GPU
     ↓
Data Parallelism
     ↓
DistributedDataParallel
     ↓
Multi-GPU Training
     ↓
Distributed Training
```

---

## 📌 How to Use These Resources

You do not need to read everything in this directory.

Use the resources according to your current objective.

If you're working on **profiling**, start with the PyTorch Profiler resources.

If you're investigating **slow data loading**, look at the DataLoader and CUDA memory-transfer documentation.

If you're working on **multi-GPU training**, start with DistributedDataParallel.

If you want to understand **GPU execution at a lower level**, move toward NVIDIA's CUDA and Nsight documentation.

---

## 🤝 Contributing Resources

This directory is open to contributions.

If you discover a useful:

- Tutorial
- Documentation page
- Research paper
- Technical article
- Benchmark
- Profiling guide
- CUDA example
- Performance engineering resource

feel free to add it.

When adding a resource, please include:

1. The resource title
2. The URL
3. A short description
4. The topic it belongs to
5. Why it is useful

Please prefer **official documentation, high-quality technical articles, and reproducible examples**.

---

## 🔎 Resource Quality

When adding resources, prioritize:

- Official NVIDIA documentation
- Official PyTorch documentation
- Official Hugging Face documentation
- Reputable technical publications
- Research papers
- Reproducible benchmarks
- Practical engineering guides

Avoid adding resources that make unsupported performance claims or provide benchmarks without explaining the methodology.

---

## 🚀 Keep Exploring

The resources here are only a starting point.

GPU performance engineering is a large field involving:

```text
PyTorch
   │
   ├── CUDA
   │
   ├── cuDNN
   │
   ├── cuBLAS
   │
   ├── Tensor Cores
   │
   ├── Memory Systems
   │
   ├── CUDA Kernels
   │
   ├── Profiling
   │
   └── Distributed Training
```

The best way to learn is to combine documentation with experiments.

**Read → Implement → Profile → Measure → Optimize → Repeat**

```

A small note on your original Sigma2 link: I converted the local `file:///...` path into the **public Sigma2 documentation URL**, since a `file://` path points to your own computer and won't work for repository users. The Sigma2 guide specifically covers PyTorch Profiler, GPU usage, kernel view, memory view, trace view, and TensorBoard visualization. :contentReference[oaicite:0]{index=0}

I also included the Hugging Face profiler article because it goes particularly well with your profiling section: it focuses on actually **reading traces**, including CPU/GPU lanes, CUDA kernels, dispatch chains, and synchronization behavior. :contentReference[oaicite:1]{index=1} The official PyTorch Profiler documentation and DDP documentation are useful as authoritative references alongside the more tutorial-oriented resources. :contentReference[oaicite:2]{index=2}
```
