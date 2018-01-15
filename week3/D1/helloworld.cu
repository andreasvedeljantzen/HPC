#include <stdio.h>
#include <helper_cuda.h>
__global__ void kernel();


int main(int argc, char **argv)
{
	// Kernel lauch
	kernel<<<4, 64>>>();

	return(0);

};

__global__ void kernel() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		printf("Hello world! Im thread %i out of %i . My Global thread id is %i out of %i \n", threadIdx.x, blockIdx.x, i, gridDim.x*blockDim.x );
	};