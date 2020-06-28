#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "assignment2.h"

int main(int argc ,char* argv[]) {
        FILE *fp;
        FILE *fp1;
	//size_t size;
        //size_t size1;
	//printf("start");  
	/* Initialize rows, cols, ncases, ncontrols from the user */
	unsigned int rows=atoi(argv[1]);
	unsigned int cols=atoi(argv[2]);
	//int CUDA_DEVICE = atoi(argv[3]);
	int nprocs = atoi(argv[3]);
	printf("Rows=%d Cols=%d  THREADS=%d\n",rows,cols,nprocs);
        
	//cudaError err = cudaSetDevice(CUDA_DEVICE);
	//if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	/*Host variable declaration */

	//int THREADS = 32;
	//int BLOCKS;
	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime, endtime;
	clock_t start, end;
	float seconds=0.0;
	unsigned int jobs; 
	unsigned long i;
	/*unsigned long ulone = 1;
	unsigned long ultwo = 2;*/ 
        //printf("print this\n");
	/*Kernel variable declaration */
	float *dev_dataT;
        float *dev_dataT1;
	float *results;
	start = clock();

	/* Validation to check if the data file is readable */
	fp = fopen(argv[4], "r");
	if (fp == NULL) {
    		printf("Cannot Open the File");
		return 0;
	}
        /*--------------------------*/
        gettimeofday(&starttime, NULL);
        float myvariable;
        float arrst[rows][cols];
        int j=0;
        for(i = 0; i < rows; i++)
        {
         for (j = 0 ; j < cols; j++)
         {
           fscanf(fp,"%f",&myvariable);
           //printf("%.15f ",myvariable);
           arrst[i][j]=myvariable;
         }
         //printf("\n");
        }
        float *dataT;
        dataT = (float*) malloc((rows*cols)*sizeof(float));
        for (i = 0; i < cols; i++) {
         for (j = 0; j < rows; j++) {
           dataT[rows*i+j]=arrst[j][i];
         }
        }
        /*for (j = 0; j < (rows*cols); j++) {
           printf("%f",dataT[j]);
         }*/
        /*-------------------------*/
	//size = (size_t)((size_t)rows * (size_t)cols);
	//printf("Size of the data = %lu\n",size);

	fflush(stdout);
        /*
	unsigned char *dataT = (unsigned char*)malloc((size_t)size);

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}

        gettimeofday(&starttime, NULL);*/

	fclose(fp);
        //printf("read data\n");
        fflush(stdout);
        //printf("Data from the file:\n%s", dataT);
        fflush(stdout);
        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        /*End of first file read and copy */

        /*vector file read and copy*/
        fp1 = fopen(argv[5], "r");
        if (fp1 == NULL) {
                printf("Cannot Open the File");
                return 0;
        }
        
        fflush(stdout);
        //unsigned char *dataT1 = (unsigned char*)malloc((size_t)size1);
        //printf("just2\n");
        float *dataT1;
        dataT1 = (float*) malloc(cols*sizeof(float));
        if(dataT1 == NULL) {
                printf("ERROR: Memory for data not allocated.\n");
        }

        gettimeofday(&starttime, NULL);

        /* Transfer the SNP Data from the file to CPU Memory */
	int v=0;
        i=0;
        for(i=0;i<cols;i++){
         fscanf(fp1,"%f",&dataT1[i]);
        }
        fclose(fp1);
        //printf("read vector data\n");
        fflush(stdout);
        //printf("Data from the file:\n%f", dataT1);
        /*i=0;
        for(i=0 ; i < cols; ++i )
            printf("%f", dataT1[i]);
        fflush(stdout);*/
        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        printf("time to read data = %f\n", seconds);
        
        /*End of vector file and copy*/
        printf("rows=%d\n", rows); 
        jobs = (unsigned int) ((rows+nprocs-1)/nprocs);
        gettimeofday(&starttime, NULL);
        printf("jobs=%d\n", jobs);
	/*Calling the kernel function */
        #pragma omp parallel num_threads(nprocs)
        kernel(rows,cols,dataT,dataT1,host_results, jobs);
	//kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,dev_dataT1,results);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
		
	/*Copy the results back in host*/
	//cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);
	printf("Output Dot Product for each row:\n");	
	
	for(v = 0; v < rows; v++) {
		printf("%f ", host_results[v]);
	}
	printf("\n");

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Total time = %f\n", seconds);

	return 0;
}
