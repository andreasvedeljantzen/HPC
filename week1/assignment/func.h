#ifndef __FUNC_LIB_H
#define __FUNC_LIB_H

/*
void matmult_lib(int m, int n, int k, double **A, double **B, double **C);
void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs);
void matmult_nat(int m, int n, int k, double **A, double **B, double **C);
*/
void matmult_mnk(int m, int n, int k, double **A, double **B, double **C);

#endif