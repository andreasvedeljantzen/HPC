/*------PI----*/
double pi_cal(int I) {
	double PI[I];
	double RealPI;
	for (int i = 1; i <= I; i++) { 
		PI[i-1] = (4/((1+(((i-0.5)/I)*((i-0.5)/I)))));
    }
    for (int i = 0; i < I; i++) { 
		RealPI += PI[i];
    }
    return(RealPI/I);
}
