#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "backend/CudaAssignmentContext.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace kmeans {
    CudaAssignmentContext::CudaAssignmentContext(int width, int height, int k) 
        : m_width(width), m_height(height), m_k(k) 
    {
        m_imgSize = width * height * 3 * sizeof(unsigned char);
        m_centersSize = k * 5 * sizeof(float);

        // Persistent memory allocation
        cudaMalloc(&d_input, m_imgSize);
        cudaMalloc(&d_output, m_imgSize);
        cudaMalloc(&d_centers, m_centersSize);

        // Initialize the stream for async operations
        cudaStreamCreate(&m_stream);
    }

    CudaAssignmentContext::~CudaAssignmentContext() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_centers) cudaFree(d_centers);
        if (m_stream) cudaStreamDestroy(m_stream);
    }

    __global__ static void assignPixelsKernel(
        const unsigned char* input,
        unsigned char* output,
        int width,
        int height,
        const float* centers,
        int k,
        float color_scale,
        float spatial_scale)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = width * height;
        if (idx >= total) return;

        int r = idx / width;
        int c = idx % width;
        int offset = idx * 3;

        float x01 = float(c) / float(width);
        float y01 = float(r) / float(height);

        float f[5] = {
            (float)input[offset + 0] * color_scale,
            (float)input[offset + 1] * color_scale,
            (float)input[offset + 2] * color_scale,
            x01 * spatial_scale,
            y01 * spatial_scale
        };

        int bestIdx = 0;
        float bestDist2 = 1e20f;

        for (int ci = 0; ci < k; ++ci) {
            float d2 = 0.0f;
            for (int d = 0; d < 5; ++d) {
                float diff = f[d] - centers[ci * 5 + d];
                d2 += diff * diff;
            }
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestIdx = ci;
            }
        }

        float inv_scale = 1.0f / fmaxf(1e-6f, color_scale);
        output[offset + 0] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 0] * inv_scale);
        output[offset + 1] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 1] * inv_scale);
        output[offset + 2] = (unsigned char)fminf(255.0f, centers[bestIdx * 5 + 2] * inv_scale);
    }

    void CudaAssignmentContext::run(const cv::Mat& frame, 
                                    const std::vector<cv::Vec<float, 5>>& centers, 
                                    cv::Mat& output,
                                    float color_scale, 
                                    float spatial_scale) 
    {
        cudaMemcpyAsync(d_input, frame.data, m_imgSize, cudaMemcpyHostToDevice, m_stream);
        
        // Flatten centers to a local vector
        std::vector<float> flatCenters;
        flatCenters.reserve(m_k * 5);
        for(const auto& c : centers) {
            for(int i = 0; i < 5; ++i) flatCenters.push_back(c[i]);
        }
        
        cudaMemcpyAsync(d_centers, flatCenters.data(), m_centersSize, cudaMemcpyHostToDevice, m_stream);

        // 2. Launch Kernel on Stream
        int threadsPerBlock = 256;
        int blocksPerGrid = (m_width * m_height + threadsPerBlock - 1) / threadsPerBlock;
        
        assignPixelsKernel<<<blocksPerGrid, threadsPerBlock, 0, m_stream>>>(
            d_input, d_output, m_width, m_height, d_centers, m_k, color_scale, spatial_scale
        );

        // 3. Async Download
        cudaMemcpyAsync(output.data, d_output, m_imgSize, cudaMemcpyDeviceToHost, m_stream);

        // 4. Synchronize the stream so 'output' is ready for OpenCV display
        cudaStreamSynchronize(m_stream);
    }
}