/*------PI----*/
/*
double pi_cal(int I) {
	double PI[I] ;
	double RealPI = 0.0;
	for (int i = 1; i <= I; i++) { 
		PI[i-1] = 4.0/(1.0+((i-0.5)/I)*((i-0.5)/I));
    }
    for (int i = 0; i < I; i++) { 
		RealPI += PI[i];
    }
    return(RealPI/I);
}*/

double pi_cal(int N) {
double arr[N];

	for (int i = 1; i <= N; i++)
		arr[i-1] = 4.0/(1.0 + ((i - 0.5)/N) * ((i - 0.5)/N));

	double PI = 0;

	for (int j = 0; j < N; j++)
		PI += arr[j];


	return(PI/N);
}
