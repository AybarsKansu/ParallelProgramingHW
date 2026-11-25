#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define CUDA_SAFE_CALL(call)                                  \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define REPS 100

#define ANGLE -1.570796f
#define NEWANGLE 1.570796f

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

using namespace std;
using namespace cv; 

// Simple transformation kernel
__global__ void transformKernel(float* output, cudaTextureObject_t texObj, int width, int height, float theta){
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<width && y<height)
	{
        float u = x / (float)width;
        float v = y / (float)height;

        // Transform coordinates
        u -= 0.5f;
        v -= 0.5f;
        float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
        float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

        // Read from texture and write to global memory
        output[y * width + x] = tex2D<float>(texObj, tu, tv);
    }
}

__global__ void transposeKernelAfterTransform(float* output, float* input, int width, int height, int out_width, int out_height){
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<out_width && y<out_height)
	{
        // Read from texture and write to global memory
        output[y * out_width + (out_width-x)] =  input[y * out_width + x];
    }
}

__global__ void transposeKernelWithTransform(float* output, cudaTextureObject_t texObj, int width, int height, int out_width, int out_height, float theta){
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<width && y<height)
	{
        float u = x / (float)width;
        float v = y / (float)height;

        // Transform coordinates
        u -= 0.5f;
        v -= 0.5f;
        float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
        float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

        // Read from texture and write to global memory
        output[(out_height-y) * width + x] = tex2D<float>(texObj, tu, tv);
    }
}

// Host code
int main()
{
   	double msec;
    std::chrono::high_resolution_clock::time_point start, end;
    
    Mat inputImg = imread("airplane.bmp", 0);
    const int height = inputImg.rows;
    const int width = inputImg.cols;
    
    const int out_width = height;
    const int out_height = width;
    
    Mat tempImg = Mat::zeros(height, width, CV_32FC1);
    Mat outputImg = Mat::zeros(out_height, out_width, CV_32FC1);
    Mat alternativeoutputImg = Mat::zeros(out_height, out_width, CV_32FC1);

    inputImg.convertTo(tempImg, CV_32FC1);

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    CUDA_SAFE_CALL(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(float);
    // Copy data located at address h_data in host memory to device memory
    CUDA_SAFE_CALL(cudaMemcpy2DToArray(cuArray, 0, 0, tempImg.data, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocate result of transformation in device memory
    float *output;
    float *newoutput;
    float *alternativeoutput;
    CUDA_SAFE_CALL(cudaMalloc(&output, width * height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&newoutput, out_width * out_height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&alternativeoutput, out_width * out_height * sizeof(float)));

    // Invoke kernel
    dim3 threadsperBlock(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 numBlocks((out_width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (out_height + threadsperBlock.y - 1) / threadsperBlock.y);
                    
    msec = 0;
	for(int c=0; c<REPS; c++) { 
		start = std::chrono::high_resolution_clock::now();
        transformKernel<<<numBlocks, threadsperBlock>>>(output, texObj, width, height, ANGLE);
        transposeKernelAfterTransform<<<numBlocks, threadsperBlock>>>(newoutput, output, width, height, out_width, out_height);
        transposeKernelWithTransform<<<numBlocks, threadsperBlock>>>(alternativeoutput, texObj, width, height, out_width, out_height, NEWANGLE);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        msec += static_cast<float>(duration) / 1000.0f;	
    }
    printf("CUDA texture rotate (no overheads) averaged over %d trials is %f msecs \n", REPS, msec/REPS);
    
    // Copy data from device back to host
    CUDA_SAFE_CALL(cudaMemcpy(outputImg.data, newoutput, out_width * out_height * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(alternativeoutputImg.data, alternativeoutput, out_width * out_height * sizeof(float), cudaMemcpyDeviceToHost));
    
    outputImg.convertTo(outputImg, CV_8UC1);
    alternativeoutputImg.convertTo(alternativeoutputImg, CV_8UC1);
    
    imshow("Input image", inputImg); 
    imshow("Output image", outputImg);
    imshow("A image", alternativeoutputImg);
    
    waitKey(0);
    destroyAllWindows();

    // Destroy texture object
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj));

    // Free device memory
    CUDA_SAFE_CALL(cudaFreeArray(cuArray));
    CUDA_SAFE_CALL(cudaFree(output));

    // Free host memory
    inputImg.release();
    tempImg.release();
    outputImg.release();
    alternativeoutputImg.release();
    
    return 0;
}
