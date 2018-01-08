/*------PI----*/
void pi_cal(int I) {
	double PI[I];
	double RealPI;
	for (int i = 1; i <= I; i++) { 
		PI[i-1] = (4/((1+pow((i-0.5)/I),2)));
    }
    for (int i = 0; i < I; i++) { 
		RealPI += PI[i];
    }
    return(RealPI);
}