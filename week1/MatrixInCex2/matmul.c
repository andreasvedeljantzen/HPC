void
matadd(int m, int n, double **A, double *B, double *C) {
    
    int i, j;
	double tempval;
	//printf("go\n");
    for(i = 0; i < m; i++)
	{
		//printf("go2\n");
		tempval = 0.0;
		for(j = 0; j < n; j++)
		{
			tempval = tempval + A[i][j] * B[j];
		}
		C[i] = tempval;
	}
}