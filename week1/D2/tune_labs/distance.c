#include "distance.h"
#include <unistd.h>
#include <math.h>

#ifdef ALL_IN_ONE

double 
distance(particle_t *p, int n) {

	for(int i = 0; i < nparts; i++ ) 

		double total_length;
		total_length = 0.0;
		{
      		p.dist[i] = sqrt(pow(p[i].x,2) + pow(p[i].y,2) + pow(p[i].z,2));
      		total_length += p.dist[i]
  		}
    //sleep(1);
    return(total_length);
}

#else

double 
distance(particle_t *p, double *v, int n) {

		double total_length;
		total_length = 0.0;
		for(int i = 0; i < nparts; i++ ) 
		{
    		v[i] = sqrt(pow(p[i].x,2) + pow(p[i].y,2) + pow(p[i].z,2));
    		total_length += v[i];
		}
    //sleep(1);
    return(total_length);
}

#endif
