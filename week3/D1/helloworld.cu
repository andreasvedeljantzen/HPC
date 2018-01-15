int main(int argc, char **argv)
{
	// Variable tid is in local memory and private to each thread
 	int tid;

	// Transfer data from host to device
	cudaMemcpy(...);

	// Kernel lauch
	kernel<<<4, 64>>>();
	__global__ void kernel() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		printf("Hello world! Im thread %ithreadIdx.x out of %iblockIdx.x . My Global thread id is %ithreadIdx out of %iGridDim.x");
	};

	cudaDeviceSynchronize();

	// Transfer results from device to host
	cudaMemcpy(...);

 	// Built-in variables like threadIdx.x are in local memory

} 