#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include "solution.h"
int main(int argc ,char* argv[]) {
        FILE *fp;
        FILE *fp1;
	//size_t size;
        //size_t size1;
	//printf("start");  
	/* Initialize rows, cols, ncases, ncontrols from the user */
	unsigned int rows=atoi(argv[1]);
	unsigned int cols=atoi(argv[2]);
	int CUDA_DEVICE = atoi(argv[3]);
	int THREADS = atoi(argv[4]);
	printf("Rows=%d Cols=%d CUDA_DEVICE=%d THREADS=%d\n",rows,cols,CUDA_DEVICE,THREADS);
        
	cudaError err = cudaSetDevice(CUDA_DEVICE);
	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	/*Host variable declaration */

	//int THREADS = 32;
	int BLOCKS;
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
        //char *line = NULL; size_t len = 0;
	//har *token, *saveptr;
        //char *line1 = NULL; size_t len1 = 0;
        //char *token1, *saveptr1;
	start = clock();

	/* Validation to check if the data file is readable */
	fp = fopen(argv[5], "r");
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

        /* Transfer the SNP Data from the file to CPU Memory 
        i=0;
	while (getline(&line, &len, fp) != -1) {
                token = strtok_r(line, " ", &saveptr);
                while(token != NULL){
                        dataT[i] = *token;
                        i++;
                        token = strtok_r(NULL, " ", &saveptr);
                }

  	}*/
	fclose(fp);
        //printf("read data\n");
        fflush(stdout);
        //printf("Data from the file:\n%s", dataT);
        fflush(stdout);
        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        //printf("time to read data = %f\n", seconds);
        /*End of first file read and copy */

        /*vector file read and copy*/
        //printf("here\n");
        fp1 = fopen(argv[6], "r");
        if (fp1 == NULL) {
                printf("Cannot Open the File");
                return 0;
        }
        //int rows1=1;
        //size1 = (size_t)((size_t)rows1 * (size_t)cols);
        //printf("Size of the vector data = %lu\n",size1);
        
        fflush(stdout);
        //printf("just1\n");
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
        //i=0;
        //printf("just\n");
        //fflush(stdout);
        //fscanf(fp1,"%[^\n]", dataT1);
        /*(while (getline(&line1, &len1, fp1) != -1) {
                token1 = strtok_r(line1, " ", &saveptr1);
                printf("now1\n");
                fflush(stdout);
                while(token1 != NULL){
                        printf("now1\n");
                        fflush(stdout);
                        dataT1[i] = *token1;
                        i++;
                        token = strtok_r(NULL, " ", &saveptr1);
                }

        }*/
        //float *B;
        //B = (float*) malloc(col*sizeof(float));

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
            //printf("%f", dataT1[i]);
            fflush(stdout);*/
        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        printf("time to read data = %f\n", seconds);

        /*End of vector file and copy*/
	/* allocate the Memory in the GPU for both data files */	   
        gettimeofday(&starttime, NULL);
	//err = cudaMalloc((unsigned char**) &dev_dataT, (size_t) size * (size_t) sizeof(unsigned char) );
        err = cudaMalloc((float**) &dev_dataT, (rows*cols) * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for cudamalloc for data file=%f\n", seconds);
        
        gettimeofday(&starttime, NULL);
        //err = cudaMalloc((unsigned char**) &dev_dataT1, (size_t) size1 * (size_t) sizeof(unsigned char) );
        err = cudaMalloc((float**) &dev_dataT1, cols * sizeof(float) );
        if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time for cudamalloc for vector file=%f\n", seconds);

        gettimeofday(&starttime, NULL);
	err = cudaMalloc((float**) &results, rows * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for cudamalloc for output results file=%f\n", seconds);

	/*Copy the SNP data to GPU */
        gettimeofday(&starttime, NULL);
	//err = cudaMemcpy(dev_dataT, dataT, (size_t)size * (size_t)sizeof(unsigned char), cudaMemcpyHostToDevice);
        err = cudaMemcpy(dev_dataT, dataT, (rows*cols)*sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time to copy data to GPU=%f\n", seconds);

        gettimeofday(&starttime, NULL);
        err = cudaMemcpy(dev_dataT1, dataT1, cols*sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time to copy vector data to GPU=%f\n", seconds);

	jobs = rows;
	BLOCKS = (jobs + THREADS - 1)/THREADS;

        gettimeofday(&starttime, NULL);
	/*Calling the kernel function */
	kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,dev_dataT1,results);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
		
	/*Copy the results back in host*/
	cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);
	printf("Output Dot Product for each row:\n");	
	for(int k = 0; k < jobs; k++) {
		printf("%f ", host_results[k]);
	}
	printf("\n");

	cudaFree( dev_dataT );
	cudaFree( results );

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Total time = %f\n", seconds);

	return 0;
}
