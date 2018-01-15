//This function is for writing a matrix into a file
void (double *U, int N, int delta)
{

	double u,y,x = 0.0;
FILE *matrix=fopen("matrix.txt", "w");

for (int i = 0; i < N-1; i++) {
	x = -1 + i* delta;
	for (int j = 0; j < N-1; j++) {
		y = -1 + j* delta;

		u = U[i*N + j]
		fprintf(matrix, "%g\t%g\t%g\n", x,y,u);

	}
}

fclose(matrix);
}
