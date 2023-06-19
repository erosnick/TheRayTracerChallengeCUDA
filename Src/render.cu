#include "render.cuh"

#include "OpenGL.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "tuple.h"

dim3 blockSize;
dim3 gridSize;

cudaArray* cudaTextureArray = nullptr;

tuple* deviceImageData = nullptr;

cudaGraphicsResource* cudaTextureResource = nullptr;

curandState* deviceRandState = nullptr;

void cudaAllocateTextureMemory(tuple** deviceImageData, size_t size);

void cudaFreeTextureMemory(tuple* deviceImageData);

void writeToPNG(const std::string& path, int32_t width, int32_t height, const std::vector<uint8_t>& pixelData);

void createCUDAResources();

void releaseCUDAResources();

void updateOpenGLTexture();

__global__ void renderInit(int width, int height, curandState* rand_state);

__global__ void cudaWriteTexture(tuple* imageData, int width, int height, curandState* randState);

// CUDA function to allocate device memory for the texture
void cudaAllocateTextureMemory(tuple** deviceImageData, size_t size)
{
    cudaMalloc((void**)deviceImageData, size);
}

// CUDA function to free device memory for the texture
void cudaFreeTextureMemory(tuple* deviceImageData)
{
    cudaFree(deviceImageData);
}

void writeToPNG(const std::string& path, int32_t width, int32_t height, const std::vector<uint8_t>& pixelData)
{
    stbi_write_png(path.c_str(), width, height, 3, pixelData.data(), width * 3);

    std::cout << "Write to " << path << std::endl;
}

void createCUDAResources()
{
    // Allocate device memory for the texture
    cudaAllocateTextureMemory(&deviceImageData, WindowWidth * WindowHeight * sizeof(tuple));

    // Set CUDA-OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTextureResource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaTextureResource, 0, 0));

    // Launch CUDA kernel to write data to the texture
    blockSize = { 16, 16 };
    gridSize = { (WindowWidth + blockSize.x - 1) / blockSize.x, (WindowHeight + blockSize.y - 1) / blockSize.y };

    checkCudaErrors(cudaMalloc((void**)&deviceRandState, WindowWidth * WindowHeight * sizeof(curandState)));

    renderInit << <gridSize, blockSize >> > (WindowWidth, WindowHeight, deviceRandState);

    updateOpenGLTexture();

    // Unmap the texture from CUDA resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTextureResource));

    // Read back the texture data from device to host
    tuple* hostImageData = new tuple[WindowWidth * WindowHeight];
    checkCudaErrors(cudaMemcpy(hostImageData, deviceImageData, WindowWidth * WindowHeight * sizeof(tuple), cudaMemcpyDeviceToHost));

    std::vector<uint8_t> pixelData;

    // Access the texture data on the CPU side
    for (int y = WindowHeight - 1; y >= 0; y--)
    {
        for (int x = 0; x < WindowWidth; ++x) 
        {
            int index = y * WindowWidth + x;
            tuple color = hostImageData[index];
            // Access individual color components: color.x, color.y, color.z, color.w
            pixelData.push_back(static_cast<uint8_t>(color.x * 255.9f));
            pixelData.push_back(static_cast<uint8_t>(color.y * 255.9f));
            pixelData.push_back(static_cast<uint8_t>(color.z * 255.9f));
        }
    }

    writeToPNG("render.png", WindowWidth, WindowHeight, pixelData);

    // Free the host memory
    delete[] hostImageData;
}

void releaseCUDAResources()
{
    cudaGraphicsUnregisterResource(cudaTextureResource);
    cudaFreeTextureMemory(deviceImageData);

    cudaFree(deviceRandState);
}

void updateOpenGLTexture()
{
    cudaWriteTexture << <gridSize, blockSize >> > (deviceImageData, WindowWidth, WindowHeight, deviceRandState);

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy CUDA texture data to OpenGL texture
    checkCudaErrors(cudaMemcpy2DToArray(cudaTextureArray, 0, 0, deviceImageData, WindowWidth * sizeof(tuple),
        WindowWidth * sizeof(tuple), WindowHeight, cudaMemcpyDeviceToDevice));
}

__global__ void renderInit(int width, int height, curandState* rand_state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixelIndex = j * width + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixelIndex, 0, &rand_state[pixelIndex]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixelIndex, 0, 0, &rand_state[pixelIndex]);
}

// CUDA kernel function for writing data to a texture
__global__ void cudaWriteTexture(tuple* imageData, int width, int height, curandState* randState)
{
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int32_t pixelIndex = y * width + x;

        curandState localRandState = randState[pixelIndex];

        float r = float(curand_uniform(&localRandState));
        float g = float(curand_uniform(&localRandState));
        float b = float(curand_uniform(&localRandState));

        tuple color;
        color.x = r;  // Red component
        color.y = g;  // Green component
        color.z = b;  // Blue component
        color.w = 1.0f;  // Alpha component

        imageData[pixelIndex] = color;

        randState[pixelIndex] = localRandState;
    }
}