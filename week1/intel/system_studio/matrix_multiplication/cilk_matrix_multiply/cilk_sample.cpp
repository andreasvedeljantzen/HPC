/*******************************************************************************
*   Copyright(C) 2012-2014 Intel Corporation. All Rights Reserved.
*   
*   The source code, information  and  material ("Material") contained herein is
*   owned  by Intel Corporation or its suppliers or licensors, and title to such
*   Material remains  with Intel Corporation  or its suppliers or licensors. The
*   Material  contains proprietary information  of  Intel or  its  suppliers and
*   licensors. The  Material is protected by worldwide copyright laws and treaty
*   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
*   modified, published, uploaded, posted, transmitted, distributed or disclosed
*   in any way  without Intel's  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*   
*   *Third Party trademarks are the property of their respective owners.
*   
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel's
*   suppliers or licensors in any way.
*
********************************************************************************/

#include <stdio.h>
#include <cilk/cilk.h>
#include "Windows.h"

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 10

int main()
{
    double *A, *B, *C;
    int m, n, p, i,r;
   	DWORD start, total;

    printf ("\n This example measures performance of  matrix product \n");

    m = 2000, p = 200, n = 1000;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, p, p, n);
       
  /*  printf (" Allocating memory for matrices \n\n"); */
			
  A = (double *)malloc( m*p*sizeof( double ));
    B = (double *)malloc( p*n*sizeof( double ));
    C = (double *)malloc( m*n*sizeof( double ));
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }
		
	//printf (" Intializing matrix data \n\n");
	
	for (i = 0; i < (m*p); i++) {
        A[i] = (double)(i+1);
    }

	for (i = 0; i < (p*n); i++) {
        B[i] = (double)(-i-1);
    }
	
    for (i = 0; i < (m*n); i++) { 
        C[i] = 0.0;
    }

    /*printf (" Making the first run of matrix product using triple nested loop\n\n"); */
    cilk_for (int i = 0; i < m; i++) {
		int j,k;
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < p; k++)
                sum += A[p*i+k] * B[n*k+j];
            C[n*i+j] = sum;
        }
    }

    //printf (" Measuring performance of matrix product using triple nested loop \n\n");
 	start = GetTickCount();
	int j,k;
    for (r = 0; r < LOOP_COUNT; r++) {
		//Cilk_for implementation
        cilk_for (int i = 0; i < m; i++) {
			int j;
            for (j = 0; j < n; j++) {
				//Array notation implementation
				/*A[p*i+k] --> A[p+0], A[p+1], A[p+2], A[p+3], A[p+4].....

				Diff/Stride = (p+1)-(p+0) = 1

				Array notation :- [<lower bound> : <length> : <stride>] 

				A[p*i:p:1]

				B[n*k+j] --> B[0+j], B[n+j], B[2n+j], B[3n+j], B[4n+j]..... 

				Diff/Stride = (n+j)-(0+j) = n

				Array notation :- [<lower bound> : <length> : <stride>] 

				B[j:p:n]
                */
				
				C[n*i+j] = __sec_reduce_add(A[p*i:p] * B[j:p:n]);
            }
        }
    }
	
	total = (GetTickCount() - start)/LOOP_COUNT;
	
	printf (" == Matrix multiplication using Cilk Plus completed == \n"
            " == at %d milliseconds == \n\n", total);

    printf (" Deallocating memory \n\n");
    free(A);
    free(B);
    free(C);
    
    printf (" Example completed. \n\n");
    return 0;
}
