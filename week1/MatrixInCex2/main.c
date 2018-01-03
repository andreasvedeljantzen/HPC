#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"		/* helper functions	        */
#include "maddmul.h"		/* my matrix add fucntion	*/

#define NREPEAT 100		/* repeat count for the experiment loop */

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {

    int    i, m, n, N = NREPEAT;
    double **A, *B, *C, *tempval;
    double tcpu1; 

    clock_t t1, t2;
	
	printf("chk1\n");

    for (m = 200; m <= 3500; m += 300) {
	n = m + 25;

	/* Allocate memory */
	A = malloc_2d(m, n);
	B = malloc(n* sizeof(double ));
	C = malloc(m* sizeof(double ));
	if (A == NULL || B == NULL | C == NULL) {
	    fprintf(stderr, "Memory allocation error...\n");
	    exit(EXIT_FAILURE);
	}

	/* initialize with useful data - last argument is reference */
	init_data(m,n,A,B);
	
	printf("chk2\n");

	/* timings for matadd */
	t1 = mytimer();
	for (i = 0; i < N; i++)
	    matadd(m, n, A, B, C);
	t2 = mytimer();
	tcpu1 = delta_t(t1, t2) / N;
	printf("chk3\n");
	/*check_results("main", m, n, C);

	/* Print n and results  */
	printf("%4d %4d %8.3f\n", m, n, tcpu1);
	if (m == 200) {
		for(i = 0; i < m; i++)
		{
			printf("%lg \n", C[i]);
		}
	}

	/* Free memory */
	free_2d(A);
	free(B);
	free(C);
    }

    return EXIT_SUCCESS;
}
