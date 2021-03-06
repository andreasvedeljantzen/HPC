//#ifdef _OPM
#include <omp.h>
//#endif _OPM

__global__ void jac_mp_v3(int N, double delta, int max_iter, double *f, double *d_u, double *d_u_old);

void write_result(double *U, int N, double delta, char filename[20]);
