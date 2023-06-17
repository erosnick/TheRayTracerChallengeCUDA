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

tuple* devImageData = nullptr;

cudaGraphicsResource* cudaTextureResource = nullptr;

void updateOpenGLTexture();

__global__ void cudaWriteTexture(tuple* imageData, int width, int height);

// CUDA function to allocate device memory for the texture
void cudaAllocateTextureMemory(tuple** devImageData, size_t size)
{
    cudaMalloc((void**)devImageData, size);
}

// CUDA function to free device memory for the texture
void cudaFreeTextureMemory(tuple* devImageData)
{
    cudaFree(devImageData);
}

void writeToPNG(const std::string& path, int32_t width, int32_t height, const std::vector<uint8_t>& pixelData)
{
    stbi_write_png(path.c_str(), width, height, 3, pixelData.data(), width * 3);

    std::cout << "Write to " << path << std::endl;
}

void createCUDAResources()
{
    // Allocate device memory for the texture
    cudaAllocateTextureMemory(&devImageData, WindowWidth * WindowHeight * sizeof(tuple));

    // Set CUDA-OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTextureResource));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaTextureResource, 0, 0));

    // Launch CUDA kernel to write data to the texture
    blockSize = { 16, 16 };
    gridSize = { (WindowWidth + blockSize.x - 1) / blockSize.x, (WindowHeight + blockSize.y - 1) / blockSize.y };

    updateOpenGLTexture();

    // Unmap the texture from CUDA resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTextureResource));

    // Read back the texture data from device to host
    tuple* hostImageData = new tuple[WindowWidth * WindowHeight];
    checkCudaErrors(cudaMemcpy(hostImageData, devImageData, WindowWidth * WindowHeight * sizeof(tuple), cudaMemcpyDeviceToHost));

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
    cudaFreeTextureMemory(devImageData);
}

void updateOpenGLTexture()
{
    cudaWriteTexture << <gridSize, blockSize >> > (devImageData, WindowWidth, WindowHeight);

    checkCudaErrors(cudaDeviceSynchronize());

    // Copy CUDA texture data to OpenGL texture
    checkCudaErrors(cudaMemcpy2DToArray(cudaTextureArray, 0, 0, devImageData, WindowWidth * sizeof(tuple),
        WindowWidth * sizeof(tuple), WindowHeight, cudaMemcpyDeviceToDevice));
}

// CUDA kernel function for writing data to a texture
__global__ void cudaWriteTexture(tuple* imageData, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int index = y * width + x;
        tuple color;
        color.x = float(x) / width;  // Red component
        color.y = float(y) / height;  // Green component
        color.z = 0.0f;  // Blue component
        color.w = 1.0f;  // Alpha component

        imageData[index] = color;
    }
}

void render()
{

}