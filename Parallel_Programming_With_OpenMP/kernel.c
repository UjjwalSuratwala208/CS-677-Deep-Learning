#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "assignment2.h"
void kernel(unsigned int rows, unsigned int cols ,float *dotdata,float *vecdata,float *results, unsigned int jobs){

	//int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int k,stop;
	int tid = omp_get_thread_num();
        float dp=0.0;
        int j;
        if((tid+1)*jobs > rows) {stop=rows;}
        else {stop = (tid+1)*jobs;}
        printf("thread id=%d, start=%d, stop=%d\n", tid, tid*jobs, stop);
	for(j = tid*jobs; j < stop; j++){
                dp=0.0;
		for(k=0;k<cols;k++){
			dp+=vecdata[(size_t)k]*dotdata[(size_t) ((size_t) k) * ((size_t) rows) + j];
		}
		results[j] = dp; 
	}
}
