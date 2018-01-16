#ifndef __MANDEL_H
#define __MANDEL_H

void __global__ mandel(int disp_width, int disp_height, int *array, int max_iter, int block);

#endif
