# Architecture & Developer Guide: Real-Time Quantum-Inspired 5D K-Means Segmentation

This guide provides an in-depth technical analysis of the application's architecture, design patterns, GPU kernel optimization techniques, and the quantum-inspired similarity metric.

---

## 1. System Architecture Overview

The system is engineered as a high-throughput, multi-threaded C++/CUDA pipeline designed to perform real-time 5D K-Means image segmentation at frame rates exceeding 120+ FPS.

### Thread Separation Model

To prevent GUI rendering latency from bottlenecking the compute pipeline (and vice versa), the application is split into two independent execution contexts:

1. **Main/UI Thread**: 
   - Manages the GLFW window context.
   - Executes the Dear ImGui render loop for real-time telemetry, configuration adjustment, and rendering.
   - Renders the segmented video frames as OpenGL textures asynchronously.
2. **Worker/Processing Thread**:
   - Captures raw video frames from the camera or media file using OpenCV.
   - Submits processing tasks to the `ClusteringManager`.
   - Executes CUDA kernels for data preprocessing, centroid optimization (learning phase), and full-frame pixel assignment.

### GPU Threading Model & Grid Configurations

Parallel execution of 5D clustering on the GPU requires mapping the 2D image coordinates to CUDA's thread and block grids.
* **StridedDataPreprocessor**: Launches a 2D grid of thread blocks where each thread is mapped directly to an output pixel in the downsampled image.
* **Thread Block Size**: A block size of \f$16 \times 16\f$ (\f$256\f$ threads per block) is selected. This layout maximizes SM (Streaming Multiprocessor) occupancy, balancing registers per thread and allowing the hardware scheduler to hide global memory latency effectively.

### GPU-Accelerated K-Means++ Initialization

K-Means++ initialization has a sequential complexity of \f$\mathcal{O}(N \cdot K)\f$, making it a CPU bottleneck. The pipeline offloads this to the GPU:
1. For each of the $K$ seeds, a parallel distance-update kernel calculates the squared distance between pixels and the latest centroid.
2. A parallel prefix sum (reduction) is executed in \f$\mathcal{O}(\log N)\f$ using the **Thrust library** to construct the cumulative probability distribution.
3. This allows selecting the next seed in parallel, reducing K-Means++ initialization time to sub-millisecond ranges.


### Thread Synchronization Bridge

A thread-safe state machine utilizes `std::scoped_lock` and atomic flags to update configuration parameters (such as stride, number of clusters $K$, and algorithm type) across thread boundaries without blocking the high-frequency visualization pipeline.

---

## 2. Memory & Throughput Optimizations

Operating at high resolutions and frame rates requires minimizing PCIe bus overhead and GPU execution latency.

### Page-Locked (Pinned) Memory & DMA

Standard pageable host memory requires the OS to copy data to a temporary page-locked buffer before initiating Direct Memory Access (DMA) transfers to the GPU. 
- The system bypasses this by pre-allocating persistent pinned host memory buffers using `cudaMallocHost` for input frames, output labels, and centroids.
- Pinned memory enables asynchronous data transfers via `cudaMemcpyAsync` scheduled on a dedicated CUDA stream (`cudaStream_t`).
- By overlapping memory transfers with kernel execution, the PCIe bus latency is effectively hidden.

### Single-Precision (FP32) Scientific Justification

While high-precision scientific computing often relies on double-precision (FP64) floating points, this pipeline is intentionally designed using single-precision (FP32) floats for three hardware-level reasons:
* **Hardware Compute Capacity**: Modern consumer GPUs feature a massive \f$64:1\f$ compute core ratio of FP32 relative to FP64. Implementing FP32 allows the system to utilize the full processing capacity of the GPU.
* **Precision Bounds**: Input pixel channels are represented as 8-bit integers (\f$0-255\f$). The 24-bit mantissa of an FP32 float is more than sufficient to normalize and cluster these ranges without convergence oscillations.
* **Memory Bandwidth Efficiency**: FP32 halves the memory transfer sizes compared to FP64, which minimizes PCIe transfer times and GPU cache pressure when shuffling millions of 5D vectors per second.

### Array of Structures (AoS) Layout

The strided preprocessor formats the BGR-XY pixel features into a contiguous 5-element array layout per pixel. This Array of Structures (AoS) ensures that threads executing concurrently in a warp access adjacent memory slots, maximizing L1 cache line hit rates and ensuring coalesced global memory access.

---

## 3. GPU Kernel Details

### Shared Memory Cooperative Loading

Pixel assignment requires evaluating distances to all $K$ centroids. Reading centroids from global VRAM for each pixel results in extreme memory bandwidth pressure:
\f[\mathcal{O}(\text{Pixels} \times K)\f] global reads.

To optimize this, both the classical and quantum-inspired kernels implement a cooperative load sequence:
1. Threads in a block cooperatively copy the current centroids from global memory to on-chip `__shared__` memory.
2. Each thread loads a chunk of the shared array, and `__syncthreads()` guarantees completion.
3. This reduces global reads to:
\f[\mathcal{O}(\text{Pixels} + K)\f]

@code{.cpp}
__global__ static void assignPixelsKernel(const unsigned char* input, unsigned char* output, 
                                          int width, int height, const float* centers, int k) {
    extern __shared__ float s_centers[];
    int tid = threadIdx.x;
    int centersCount = k * 5;
    for (int i = tid; i < centersCount; i += blockDim.x) {
        s_centers[i] = centers[i];
    }
    __syncthreads();
    // ... pixel assignment math follows using s_centers ...
}
@endcode

### Two-Tier Atomic Reduction

During centroid re-calculation, summing up the values of coordinates assigned to each cluster causes severe serialization due to VRAM memory contention. The system implements a two-tier reduction model:
- **Tier 1 (Block Level)**: Individual threads aggregate pixel sums and counts within fast `__shared__` memory using `atomicAdd`.
- **Tier 2 (Global Level)**: A single thread per block writes the block's sub-totals to global VRAM, reducing memory contention by a factor of 256.

---

## 4. Quantum-Inspired Distance Metric

The core mathematical innovation of this codebase is the **Quantum-Inspired Cosine Similarity Metric** (simulating a quantum SWAP test) in a 5D Hilbert phase space.

### Born's Rule and the SWAP Test

In quantum computing, the SWAP test measures the overlap (inner product) of two quantum states \f$|\psi\rangle\f$ and \f$|\phi\rangle\f$ by evaluating the probability of measuring an ancillary qubit in the \f$|0\rangle\f$ state:
\f[P(|0\rangle) = \frac{1}{2} + \frac{1}{2}| \langle\psi|\phi\rangle |^2\f]

If the states are identical, \f$P(|0\rangle) = 1.0\f$. If they are orthogonal, \f$P(|0\rangle) = 0.5\f$.

### Hilbert Phase Space Mapping

The 5D features (BGR spatial-XY) are normalized to \f$[0, 1]\f$ and mapped to angular phase orientations in a bounded domain:
\f[\theta_d = (p_d - c_d) \cdot \frac{\pi}{4}\f]

Evaluating multi-qubit product state overlaps on the GPU is approximated by:
\f[D_{\text{Quantum}}(\mathbf{p}, \mathbf{c}) = 1.0 - \prod_{d=1}^{5} \cos^2\left( (p_d - c_d) \cdot \theta_{\text{offset}} \right)\f]

This metric generates steep gradients near cluster boundaries, which helps suppress noise and sharpens edge segmentation.

### Hardware Acceleration (SFU Intrinsics)

Evaluating trigonometric functions (like cosine) can be slow on standard CUDA cores. The `QuantumEngine` compiles these operations to NVIDIA **Special Function Units (SFUs)** by utilizing the CUDA hardware intrinsic `__cosf()`:
- `__cosf()` executes in a fraction of the clock cycles required by the standard compiler math functions, maintaining execution speed parity with classical Euclidean distance calculation.
