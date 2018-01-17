#ifdef _OPM
#include <omp.h>
#endif _OPM

__global__ int jac_mp_v3(int N, double delta, double threshold, int max_iter, double *f, double *u, double *u_old);

void write_result(double *U, int N, double delta, char filename[20]);
