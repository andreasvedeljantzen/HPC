#include "distance.h"
#include <unistd.h>
#include <math.h>

#ifdef ALL_IN_ONE

double 
distance(particle_t *p, int n) {
	
	double total_length;
	total_length = 0.0;
	for(int i = 0; i < n; i++ ) 
		{
      		p[i].dist = sqrt(pow(p[i].x,2) + pow(p[i].y,2) + pow(p[i].z,2));
      		total_length += p[i].dist;
  		}
    //sleep(1);
    return(total_length);
}

#else

double 
distance(particle_t *p, double *v, int n) {

		double total_length;
		total_length = 0.0;
		for(int i = 0; i < n; i++ ) 
		{
    		v[i] = sqrt(pow(p[i].x,2) + pow(p[i].y,2) + pow(p[i].z,2));
    		total_length += v[i];
		}
    //sleep(1);
    return(total_length);
}

#endif
