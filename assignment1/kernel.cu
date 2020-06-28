#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "solution.h"
__global__ void kernel(unsigned int rows, unsigned int cols ,float *dotdata,float *vecdata,float *results){
int tid  = threadIdx.x + blockIdx.x * blockDim.x;
//printf("%i",tid);
//char *test = "12.11";
//int j;
/*for (j = 0; j < (rows*cols); j++) {
           printf("%f",dotdata[j]);
         }*/
//printf("\n");
float dp=0.0;
int i=0;
for(i=0;i<cols;i++){
 //printf("job1\n");
 //if(tid==3){
 //printf("%f\n",dotdata[i*rows+tid]);}
 dp+=vecdata[i]*dotdata[i*rows+tid];
 //if(tid==3){
 //printf("%f\n",dp);}
}
results[tid] = dp;



}
