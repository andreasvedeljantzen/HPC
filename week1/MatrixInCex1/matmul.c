void
matadd(int m, int n, double **A, double *B, double*C, double *tempval) {
    
    int i, j;

    for(i = 0; i < m; i++)
		C[i]
		for(j = 0; j < n; j++)
 
			tempval[i] = A[i][j] * B[i];
			tempval[i] = tempval[i] + tempval[i-1];
		
}