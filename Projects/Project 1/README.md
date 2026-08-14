# Project 1 — CNN Acceleration with CUDA and PyTorch

A hands-on project focused on **accelerating a convolutional neural network training pipeline with CUDA and PyTorch**.

Starting from a CPU baseline, this project progressively introduces GPU acceleration and several optimization techniques. Each optimization is benchmarked so that you can measure its real impact rather than simply assuming that a technique will make the workload faster.

> **Measure → Profile → Optimize → Benchmark**

---

## 🎯 Project Objective

The objective of this project is to take a basic CNN training pipeline and progressively improve its performance using CUDA-aware PyTorch techniques.

You will start with a CPU implementation, move the computation to the GPU, identify performance bottlenecks, and then optimize the training pipeline step by step.

By the end of the project, you should be able to explain **where the time is spent, what limits GPU utilization, which optimizations help, and why they help**.

---

## 🧠 What You'll Explore

Throughout the project, you'll investigate several sources of performance overhead:

* CPU vs GPU computation
* Host-to-Device data transfers
* Pinned memory
* Asynchronous data transfers
* Batch-size selection
* Multiprocess DataLoader
* Automatic Mixed Precision
* GPU memory usage
* CUDA execution
* PyTorch profiling
* Performance benchmarking

The goal is not to apply every optimization blindly, but to understand the relationship between the **optimization, the bottleneck, and the measured performance improvement**.

---

## 🗺️ Project Roadmap

The project follows a progressive optimization workflow.

```text
CPU Baseline
     │
     ▼
GPU Acceleration
     │
     ▼
Host → Device Transfers
     │
     ▼
Pinned Memory
     │
     ▼
Asynchronous Transfers
     │
     ▼
Batch Size Optimization
     │
     ▼
Multiprocess Data Loading
     │
     ▼
Mixed Precision / AMP
     │
     ▼
Profiling
     │
     ▼
Final Benchmark
```

Each stage builds on the previous one.

---

## 📓 Notebook

The main entry point is the project notebook:

```text
notebook.ipynb
```

The notebook contains the experiments, implementations, benchmarks, and performance analysis.

You can run it using:

* Google Colab
* Kaggle
* A local CUDA-enabled environment
* VS Code with a CUDA-enabled Python environment
* Jupyter Notebook

A NVIDIA GPU is recommended for the complete project.

---

# 🔹 Part 1 — CPU Baseline

We begin with a standard CNN training pipeline running entirely on the CPU.

The purpose of this section is to establish a **baseline measurement**.

Before optimizing anything, we need to know how long the original implementation takes.

You will measure the training time and use it as the reference point for the following experiments.

> **Question:** If we don't have a baseline, how can we prove that an optimization actually improved performance?

---

# 🔹 Part 2 — First GPU Acceleration

Next, we move the CNN and its data to the GPU.

This introduces the fundamental PyTorch CUDA workflow:

```python
model.to(device)
data.to(device)
target.to(device)
```

You'll compare the GPU implementation against the CPU baseline and calculate the resulting speedup.

This experiment demonstrates an important principle:

> Moving computation to the GPU can provide a significant speedup, but it does not automatically mean that the GPU is being used efficiently.

---

# 🔹 Part 3 — Host-to-Device Transfer Bottleneck

Once computation is moved to the GPU, a new bottleneck can appear.

The dataset is still initially stored in CPU memory, so every batch must be transferred from **Host memory → Device memory**.

If the GPU finishes computing faster than the CPU can provide new data, the GPU may spend time waiting.

This section focuses on understanding that bottleneck.

---

# 🔹 Part 4 — Pinned Memory

We then introduce **pinned memory** through the PyTorch DataLoader.

Pinned, or page-locked, CPU memory can make transfers between CPU and GPU more efficient and enables more effective asynchronous transfers.

You'll benchmark the difference between the basic GPU pipeline and the pinned-memory version.

---

# 🔹 Part 5 — Asynchronous Data Transfer

Next, we combine pinned memory with asynchronous transfers.

Instead of relying entirely on synchronous CPU-to-GPU transfers, we use PyTorch's `non_blocking` mechanism where appropriate.

You'll measure whether this reduces the time spent waiting for data.

This is an important step toward overlapping **data movement and computation**.

---

# 🔹 Part 6 — Batch Size

Batch size has a significant impact on GPU performance.

You'll benchmark several batch sizes and observe how changing the amount of work processed by each iteration affects total training time.

The goal is not simply to choose the largest possible batch size.

You'll also consider:

* GPU memory capacity
* GPU utilization
* number of iterations
* kernel-launch overhead
* training behavior

---

# 🔹 Part 7 — Multiprocess Data Loading

The next experiment focuses on the CPU side of the input pipeline.

We'll vary the number of DataLoader workers and investigate whether additional CPU processes can prepare data quickly enough to keep the GPU busy.

You'll compare different values of `num_workers` and analyze the resulting training times.

This demonstrates an important lesson:

> More workers do not automatically mean better performance.

The optimal configuration depends on the workload and the hardware.

---

# 🔹 Part 8 — Automatic Mixed Precision

We then introduce **Automatic Mixed Precision**, or AMP.

Instead of performing every operation using FP32, compatible operations can use lower precision such as FP16.

This can reduce memory usage and take advantage of NVIDIA Tensor Cores on compatible GPUs.

You'll compare:

* FP32
* Autocast
* Full AMP with gradient scaling

and analyze both execution time and memory consumption.

---

# 🔹 Part 9 — GPU Memory

Performance isn't only about execution time.

You'll also inspect GPU memory usage and investigate what happens when tensors are allocated, deleted, and released from the CUDA caching allocator.

This section helps explain the difference between:

* allocated memory
* reserved memory
* available memory

and why deleting a tensor does not necessarily mean that CUDA immediately returns all memory to the operating system.

---

# 🔬 Part 10 — Performance Profiling

Finally, you'll use profiling to look inside the training pipeline.

Instead of only asking:

> "How long did training take?"

we can ask:

> "Where exactly is that time being spent?"

Using the PyTorch Profiler, you'll inspect CPU operations, CUDA kernels, memory activity, and synchronization.

The profiler helps us move from **benchmarking** to **diagnosing**.

---

# 📊 Final Benchmark

At the end of the project, you'll compare the different implementations.

For example:

```text
CPU
 ↓
GPU
 ↓
GPU + Pinned Memory
 ↓
GPU + Pinned Memory + Async Transfer
 ↓
Optimized Batch Size
 ↓
Optimized DataLoader
 ↓
AMP
```

The exact performance will depend on your hardware.

Your objective is therefore not to reproduce a specific number, but to **measure and explain the performance evolution on your own system**.

---

# 📈 What You Should Analyze

For every optimization, try to answer:

1. What bottleneck are we trying to solve?
2. What changed in the implementation?
3. Why should this change improve performance?
4. What happened in the benchmark?
5. How much speedup did we obtain?
6. Did memory usage change?
7. Did the optimization introduce any trade-offs?
8. Is the improvement consistent across multiple runs?

---

# 📝 Project Deliverables

Your final submission should contain:

### 1. Notebook

A completed and clearly organized version of the project notebook.

### 2. Benchmark Results

Include a comparison of the different configurations you tested.

For example:

| Configuration        | Training Time | Speedup | GPU Memory |
| -------------------- | ------------: | ------: | ---------: |
| CPU                  |             — |   1.00× |          — |
| GPU                  |             — |       — |          — |
| GPU + Pinned Memory  |             — |       — |          — |
| GPU + Async Transfer |             — |       — |          — |
| Optimized Batch Size |             — |       — |          — |
| AMP                  |             — |       — |          — |

Use the results from your own hardware.

### 3. Performance Analysis

Briefly explain which optimizations produced the largest improvements and why.

### 4. Profiling Results

Include relevant profiler observations, screenshots, or exported results.

---

# 🤝 Contribution Challenge

This project is intentionally designed to be extended.

After completing the notebook, you're encouraged to improve the implementation and contribute your work to the repository.

Possible contributions include:

* Refactoring the notebook into modular Python files
* Improving the training pipeline
* Adding better benchmarking utilities
* Adding new profiling experiments
* Testing different GPU architectures
* Comparing FP32, FP16, and BF16
* Experimenting with `torch.compile`
* Investigating `channels_last`
* Exploring CUDA Graphs
* Improving DataLoader prefetching
* Adding inference benchmarks
* Improving visualizations
* Adding documentation

If you find a meaningful optimization, **benchmark it and submit a Pull Request**.

A good contribution should explain what was changed, why it should improve performance, and provide before/after measurements.

---

# 💡 The Main Challenge

Don't just ask:

> **"How can I make this code faster?"**

Instead, ask:

> **"Where is the bottleneck, and what evidence do I have?"**

That mindset is the main objective of this project.

---

## 🏁 Final Goal

By completing this project, you should be comfortable taking a basic PyTorch training pipeline, moving it to CUDA, profiling its execution, identifying bottlenecks, applying appropriate optimizations, and validating the results with reproducible benchmarks.

The final goal isn't simply a faster CNN.

It's learning how to **systematically analyze and optimize AI workloads on GPUs**.
