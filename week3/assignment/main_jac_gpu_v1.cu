#include <stdio.h>
#include <stdlib.h>
#include "func.h"
#include <omp.h>

// setting GPU device
const int device = 0;
#define BLOCK_SIZE 16


__global__ void jac_mp_v3(int N, double delta, int max_iter, double *f, double *d_u, double *d_u_old) {
	//double *temp;	
	//temp = d_u;
	//d_u = d_u_old;
	//d_u_old = temp;
	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1; j++) {
			// Update u
			d_u[i*N + j] = 0.25 * (d_u_old[(i-1)*N + j] + d_u_old[(i+1)*N + j] + d_u_old[i*N + (j-1)] + d_u_old[i*N + (j+1)] + delta*delta*f[i*N + j]);
			}
		}
	}

int main(int argc, char *argv[]) {

    //setting up device
    cudaSetDevice(device);

    // timing
    double ts, te, mflops, memory,flop;
    int max_iter, loops, N;
        
    // command line argument sets the dimensions of the image
    if (argc == 4 ) {
        N = atoi(argv[1]) + 2;
        max_iter = atoi(argv[2]);
    }
    else {
        // use default N
        N = 32 + 2;
        max_iter = 100;
    }

    loops = 1;

    // arrays
    double   *d_f, *d_u, *d_u_old;
    double   *h_f, *h_u, *h_u_old;
    int size_f = sizeof(double)*N*N;
    int size_u = sizeof(double)*N*N;
    int size_u_old = sizeof(double)*N*N;

    // GPU
    // Allocate memory on host and device
    cudaMalloc((void**)&d_f, size_f);
    cudaMalloc((void**)&d_u, size_u);
    cudaMalloc((void**)&d_u_old, size_u_old);
    //h_f = (double*)malloc(size_f);
    cudaMallocHost((void**)&h_f, size_f);
    cudaMallocHost((void**)&h_u, size_u);
    cudaMallocHost((void**)&h_u_old, size_u_old);
    
    if (d_f == NULL || d_u == NULL || d_u_old ==NULL) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }
    if (h_f == NULL || h_u == NULL || h_u_old ==NULL) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

    double delta = 2.0/N;

    int i,j;
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            if (i >= N * 0.5  &&  i <= N * 2.0/3.0  &&  j >= N * 1.0/6.0  &&  j <= N * 1.0/3.0)
                h_f[i*N + j] = 200.0;
            else
                h_f[i*N + j] = 0.0; 

            if (i == (N - 1) || i == 0 || j == (N - 1)){
                h_u[i*N + j] = 20.0;
                h_u_old[i*N + j] = 20.0;
            }
            else{
                h_u[i*N + j] = 0.0;
                h_u_old[i*N + j] = 0.0;
            } 
        }
    }
    
    cudaMemcpy(d_f, h_f, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, size_u, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_old, h_u_old, size_u_old, cudaMemcpyHostToDevice);
    
    // do program
    ts = omp_get_wtime();
    int k;
    k=0;
    //double *temp;
    while (k < max_iter) {
	//Set u_old = u
	//temp = h_u;
	//h_u = h_u_old;
	//h_u_old = temp;
	cudaMemcpy(d_u, h_u, size_u, cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_old, h_u_old, size_u_old, cudaMemcpyHostToDevice);
        jac_mp_v3<<<1,1>>>(N, delta, max_iter,d_f,d_u,d_u_old);
	cudaMemcpy(h_u_old, d_u, size_u_old, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_u, d_u_old, size_u, cudaMemcpyDeviceToHost);
	k++;
	}

    cudaDeviceSynchronize();
    te = omp_get_wtime() - ts;
    // end program

    // Copy result back to host
    cudaMemcpy(h_u_old, d_u_old, size_u_old, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_u, d_u, size_u, cudaMemcpyDeviceToHost);
    //time_end = omp_get_wtime();
    
    for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++) {
	    printf("%g\t", h_u[i*N+j]);

}
printf("\n");
}

    // flops
    //flop=max_iter * (double)(N-2) * (double)(N-2) * 10.0;

    // calculate mflops in O
    //mflops  = flop * 1.0e-06 * loops / te;
    //memory  = 3.0 * (double)(N-2) * (double)(N-2) * sizeof(double);
    
    //printf("%d\t", n_cores);
    //printf("%g\t", memory);
    //printf("%g\t", mflops);
    //printf("%g\n", te / loops);

    // Cleanup
    cudaFreeHost(h_f);
    cudaFreeHost(h_u);
    cudaFreeHost(h_u_old); 
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_u_old);
 
    // end program
    return(0);
}

