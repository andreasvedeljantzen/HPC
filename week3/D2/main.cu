#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "writepng.h"
#include <omp.h>



int main(int argc, char *argv[]) {

    int   width, height;
    int	  max_iter;
    int   *image;

    width    = 2601;
    height   = 2601;
    max_iter = 400; 

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);

    image = (int *)cudaMalloc( width * height * sizeof(int));
    if ( image == NULL ) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

    const int device = 0;
    cudaSetDevice(device);

    mandel(width, height, image, max_iter)<<<1, 1>>>();

    cudaDeviceSynchronize();

    cudaMemcpy()

    writepng("mandelbrot.png", image, width, height);

    cudaFree();

    return(0);
}
