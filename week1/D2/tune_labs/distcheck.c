#include "distcheck.h"
#include <unistd.h>

#ifdef ALL_IN_ONE

double 
distcheck(particle_t *p, int n) {
    		double total_length;
		total_length = 0.0;
		{
      		total_length += p.dist[i]
  		}
    //sleep(1);
    return(total_length);
}

#else

double 
distcheck(double *v, int n) {
    	double total_length;
		total_length = 0.0;
		for(int i = 0; i < nparts; i++ ) 
		{
    		total_length += v[i];
		}
    //sleep(1);
    return(total_length);
}

#endif
