
/*------lib----*/
/* void matmult_lib(int M, int N, int K, double **A, double **B, double **C) {
    for (int m = 0; m < M; m++) {
    	for (int n = 0; n < N; n++) {
			C[m][n] = 0;
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*------blk----*/
/*void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs) {
    for(int m = 0; m < M; m++) {
    	for(int n = 0; n < N; n++) {
			C[m][n] = 0;
			for(int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*------nat----*/
/*			DONE			*/
/*void matmult_nat(int M, int N, int K, double **A, double **B, double **C) {
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			C[m][n] = 0;
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*-----------------------Permutations---------------------*/

/*------mnk----*/
void matmult_mnk(int M, int N, int K, double **A, double **B, double **C) {
    for (int m = 0; m < M; m++) {
    	for (int n = 0; n < N; n++) {
			C[m][n] = 0;
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

/*------mkn----*/
/*void matmult_mkn(int M, int N, int K, double **A, double **B, double **C) {
    for (int m = 0; m < M; m++) {
    	for (int k = 0; k < K; n++) {
			C[m][n] = 0;
			for (int n = 0; n < N; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*------nmk----*/
/*void matmult_nmk(int M, int N, int K, double **A, double **B, double **C) {
    for (int n = 0; n < N; m++) {
    	for (int m = 0; m < M; n++) {
			C[m][n] = 0;
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*------nkm----*/
/*void matmult_nkm(int M, int N, int K, double **A, double **B, double **C) {
    for (int n = 0; n < N; m++) {
    	for (int k = 0; k < K; n++) {
			C[m][n] = 0;
			for (int m = 0; m < M; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*------kmn----*/
/*void matmult_kmn(int M, int N, int K, double **A, double **B, double **C) {
    for (int k = 0; k < K; m++) {
    	for (int m = 0; m < M; n++) {
			C[m][n] = 0;
			for (int n = 0; n < N; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/

/*------knm----*/
/*void matmult_knm(int M, int N, int K, double **A, double **B, double **C) {
    for (int k = 0; k < K; m++) {
    	for (int n = 0; n < N; n++) {
			C[m][n] = 0;
			for (int m = 0; m < M; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}*/