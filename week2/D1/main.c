#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "func.h"

int
main(int argc, char *argv[]) {

    int    I;
    I = 1000000;

    double PI = pi_cal(I);
	    
	printf("Pi = %g\n", PI);

    return EXIT_SUCCESS;
}
