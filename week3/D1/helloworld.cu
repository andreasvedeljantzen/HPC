#include <stdio.h>
#include <helper_cuda.h>


__global__ void kernel() {
		i = blockIdx.x * blockDim.x + threadIdx.x;
		printf("Hello world! Im thread %i out of %i . My Global thread id is %i out of %i \n", threadIdx.x, blockIdx.x, i, gridDim.x*blockDim.x );
	};

int main(int argc, char **argv)
{
	const int device = 0;
	cudaSetDevice(device); 

	// Kernel lauch

	kernel<<<4, 64>>>();

	cudaDeviceSynchronize();

	return(0);

};

