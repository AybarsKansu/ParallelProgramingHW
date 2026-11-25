#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
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

#define FACTOR 1.7

using namespace std;
using namespace cv;

typedef unsigned char uchar; 

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

__global__ void textureResizeKernel(float* output,  //output surface we WRITE to... remember you can't write to textures. 
                                  cudaTextureObject_t texObj,
	                              int width,              //output width
								  int height,             //output height
								  float factor)           //scale factor
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x; 
	const int y = blockIdx.y * blockDim.y + threadIdx.y; 

	if(x<width && y<height)
	{
		// warped coordinates to sample from input  
		float normalized_x = ((float)x) / factor + 0.5f;
		float normalized_y = ((float)y) / factor + 0.5f;

		// Read from texture and write to global memory
		output[y * width + x] = tex2D<float>(texObj, normalized_x, normalized_y);
	}
}

extern "C" void cudaTexResize(float* inputImage, float* outputImage, int width, int height, float factor, double* msecLocal)
{
    std::chrono::high_resolution_clock::time_point startLocal, endLocal;
    
	int outputWidth = (int)(width * factor);
	int outputHeight = (int)(height * factor);

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    CUDA_SAFE_CALL(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(float);
    // Copy data located at address h_data in host memory to device memory
    CUDA_SAFE_CALL(cudaMemcpy2DToArray(cuArray, 0, 0, inputImage, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice));

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
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Allocate result of transformation in device memory
    float *output;
    CUDA_SAFE_CALL(cudaMalloc(&output, outputWidth * outputHeight * sizeof(float)));
    
	// Invoke kernel
	dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 dimGrid((outputWidth + dimBlock.x - 1) / dimBlock.x, (outputHeight + dimBlock.y - 1) / dimBlock.y);
	
	startLocal = std::chrono::high_resolution_clock::now();
	textureResizeKernel<<<dimGrid, dimBlock>>>(output, texObj, outputWidth, outputHeight, factor);
	cudaDeviceSynchronize();
	endLocal = std::chrono::high_resolution_clock::now();
	auto durationLocal = std::chrono::duration_cast<std::chrono::microseconds>(endLocal - startLocal).count();
    *msecLocal += static_cast<float>(durationLocal) / 1000.0f;	

    // Copy back
    CUDA_SAFE_CALL(cudaMemcpy(outputImage, output, outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost));
	
	// Destroy texture object
    CUDA_SAFE_CALL(cudaDestroyTextureObject(texObj));
    
	// Free device memory
	CUDA_SAFE_CALL(cudaFreeArray(cuArray));
	CUDA_SAFE_CALL(cudaFree(output));
}

void ResizeImage(uchar* Source, uchar* Target, int sx, int sy, float factor) //this is a shit implementation. guaranteed to work slow...
{
    int outputWidth = (int)(factor*sx) ;
    int outputHeight = (int)(factor*sy) ;
        
    for(int Y=0; Y<outputHeight; Y++)
	{
		float currY = ((float)Y)/factor;
		int nnY = floor(currY);
		float alphaY = currY - (float)nnY;
		
		for(int X=0; X<outputWidth; X++)
		{
			float currX = ((float)X)/(float)factor;
			int nnX = floor(currX);
			float alphaX = currX - (float)nnX;
			
			int OffsetIn = nnY*sx+nnX;
			int offsetOut = Y*(factor*sx)+X;
			
			Target[offsetOut] = ((float)Source[OffsetIn])*(1-alphaX)*(1-alphaY)
                               +((float)Source[OffsetIn+1])*(alphaX)*(1-alphaY)
                               +((float)Source[OffsetIn+sx+1])*(alphaX)*(alphaY)
                               +((float)Source[OffsetIn+sx])*(1-alphaX)*alphaY;						
		}
	}
}

int main()
{   
	double msec, msecLocal;
    std::chrono::high_resolution_clock::time_point start, end;
    
    Mat inputImg = imread("test.bmp", 0);
    const int height = inputImg.rows;
    const int width = inputImg.cols;
    
    Mat tempImg = Mat::zeros(width, height, CV_32FC1);
    inputImg.convertTo(tempImg, CV_32FC1);
   
    //CPU (host) size allocations
    int outputWidth = (int)(inputImg.cols * FACTOR);
    int outputHeight = (int)(inputImg.rows * FACTOR);
    Mat outputImg = Mat::zeros(outputWidth, outputHeight, CV_32FC1);
	
	Mat outputImg_CPU1 = Mat::zeros(outputWidth, outputHeight, CV_8UC1);
    Mat outputImg_CPU2 = Mat::zeros(outputWidth, outputHeight, CV_8UC1);
    Mat outputImg_GPU = Mat::zeros(outputWidth, outputHeight, CV_8UC1);

	//run CPU code - manual
	printf("Running on CPU. \n");
	msec = 0;
	for(int c=0; c<REPS; c++) { 
		start = std::chrono::high_resolution_clock::now();
	    ResizeImage(inputImg.data, outputImg_CPU1.data, inputImg.cols, inputImg.rows, FACTOR);
	    end = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        msec += static_cast<float>(duration) / 1000.0f;	
    }
    printf("Manual resize averaged over %d trials is %f msecs \n", REPS, msec/REPS);
		
	printf("CPU done! \n");
    imwrite("output_CPU1.jpg", outputImg_CPU1);
    
    
    //run CPU code - OpenCV
	printf("Running on CPU. \n");
	msec = 0;
	for(int c=0; c<REPS; c++) { 
		start = std::chrono::high_resolution_clock::now();	
	    resize(inputImg, outputImg_CPU2, Size(outputWidth, outputHeight), 0, 0, INTER_LINEAR);
	    end = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        msec += static_cast<float>(duration) / 1000.0f;	
    }
    printf("OpenCV resize averaged over %d trials is %f msecs \n", REPS, msec/REPS);
    
	printf("CPU done! \n");
    imwrite("output_CPU2.jpg", outputImg_CPU2);


	//run GPU code
	printf("Running on GPU. \n");
	
	//warm-up
	cudaTexResize((float *)tempImg.data, (float *)outputImg.data, inputImg.cols, inputImg.rows, FACTOR, &msecLocal);
	
	msec = 0;
    msecLocal = 0;
	for(int c=0; c<REPS; c++) { 
		start = std::chrono::high_resolution_clock::now();
	    cudaTexResize((float *)tempImg.data, (float *)outputImg.data, inputImg.cols, inputImg.rows, FACTOR, &msecLocal);
	    end = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        msec += static_cast<float>(duration) / 1000.0f;	
    }
    printf("CUDA texture resize (with overheads) averaged over %d trials is %f msecs \n", REPS, msec/REPS);
    printf("CUDA texture resize (no overheads) averaged over %d trials is %f msecs \n", REPS, msecLocal/REPS);
	
	outputImg.convertTo(outputImg_GPU, CV_8UC1);
	printf("GPU done! \n");
    imwrite("output_GPU.jpg", outputImg_GPU);


    imshow("Input image", inputImg); 
    imshow("CPU output 1", outputImg_CPU1);
    imshow("CPU output 2", outputImg_CPU2);
    imshow("GPU output", outputImg_GPU);
    
    waitKey(0);
    destroyAllWindows();
    
    inputImg.release();
    tempImg.release();
    outputImg.release();
    outputImg_CPU1.release();
    outputImg_CPU2.release();
    outputImg_GPU.release();

	exit(0);
}
