/* Preprocessor directives */
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1024*1024 // 
#define BLOCK_SIZE 512
#define SIGMA 1


__device__ int device_min(int a, int b) {
	return (a < b) ? a : b;
}

void gaussian_kernel(float* kernel, int kernel_size, float sigma) {
    int radius = kernel_size / 2;
    float sum = 0.0;
    for (int i = -radius; i < radius; i++) {
        value = exp(-0.5 * (i * i) / (sigma * sigma)) / (sqrt(2 * M_PI) * sigma)
        kernel[i + radius] += value;
        sum += value;
    }
}

/* CUDA kernel for convolution */
__global__ void conv_kernel(float* input, float* kernel, float* output, int input_size, int kernel_size) {

	// Find the starting index and step size for the loops
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int tstep = blockDim.x * gridDim.x;

	int output_length = input_size + kernel_size - 1;

	for (int i = tid; i < output_length; i += tstep) {

		float conv_sum = 0;

		int max_kernel_size = device_min(input_size, i);
        for (int j = 0; j < max_kernel_size; j++) {
            conv_sum += input[j] * kernel[i - j];
        }

		output[i] = conv_sum;
	}
}


int main() {
    const int inputLength = 10;
    const int kernelLength = 3;
    const int resultLength = inputLength + kernelLength - 1;

    float* inputArray = (float*)malloc(inputLength * sizeof(float));
    float* kernelArray = (float*)malloc(kernelLength * sizeof(float));
    float* resultArray = (float*)malloc(resultLength * sizeof(float));
    float* d_inputArray, * d_kernelArray, * d_resultArray;

    // Initialize input and kernel arrays with random values 
    srand(time(NULL));
    for (int i = 0; i < inputLength; ++i) {
        inputArray[i] = (float)rand() / RAND_MAX;

    }
    gaussian_kernel(kernelArray, kernelLength, SIGMA);

    // Allocate memory on GPU 
    cudaMalloc(&d_inputArray, inputLength * sizeof(float));
    cudaMalloc(&d_kernelArray, kernelLength * sizeof(float));
    cudaMalloc(&d_resultArray, resultLength * sizeof(float));

    // Copy data from host to device 
    cudaMemcpy(d_inputArray, inputArray, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernelArray, kernelArray, kernelLength * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution 
    conv_kernel(d_inputArray, d_kernelArray, d_resultArray, inputLength, kernelLength);

    // Copy result back to host 
    cudaMemcpy(resultArray, d_resultArray, resultLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the result 
    printf("Input Array: ");
    for (int i = 0; i < inputLength; ++i) {
        printf("%f ", inputArray[i]);
    }
    printf("\n");

    printf("Kernel Array: ");
    for (int i = 0; i < kernelLength; ++i) {
        printf("%f ", kernelArray[i]);
    }
    printf("\n");

    printf("Result Array: ");
    for (int i = 0; i < resultLength; ++i) {
        printf("%f ", resultArray[i]);
    }
    printf("\n");

    // Free memory 
    free(inputArray);
    free(kernelArray);
    free(resultArray);
    cudaFree(d_inputArray);
    cudaFree(d_kernelArray);
    cudaFree(d_resultArray);

    return 0;
}
