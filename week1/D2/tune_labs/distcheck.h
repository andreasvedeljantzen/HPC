#ifndef __DISTCHECK_H
#define __DISTCHECK_H

#include "data.h"

#ifdef ALL_IN_ONE
double distcheck(particle_t *, int);
#else
double distcheck(double *, int);
#endif

#define CHECK_FLOP 10	// 4(sqrt) + 3(power) + 2(addition) + 1 p.dist[i] = 10
#endif
