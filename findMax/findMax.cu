// Vector Dot Product A.B
// compile with the following command:
//
// (for GTX970)
// nvcc -arch=compute_52 -code=sm_52,sm_52 -O2 -m64 -o vecAdd vecAdd.cu
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* d_A;   // device vectors
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecDot(const float* A, float* C, int N)
{
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x; //global thread ID
    int cacheIndex = threadIdx.x; //local thread ID

    float temp = 0.0;  // register for each thread
    while (i < N) {
        temp = fmaxf(temp, A[i]);
        i += blockDim.x*gridDim.x;  
    }
   
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x/2;
    while (ib != 0) {
      if(cacheIndex < ib)
        cache[cacheIndex] = fmaxf(cache[cacheIndex], cache[cacheIndex + ib]); 

      __syncthreads();

      ib /=2;
    }
    
    if(cacheIndex == 0)
      C[blockIdx.x] = cache[0];

}


// Host code

//int main(void)
int opt(int gpuid, int N, int threadsPerBlock, int blocksPerGrid, std::ofstream& myfile)
{

	 printf("%d\n",threadsPerBlock);
	 printf("%d\n",blocksPerGrid);
    // Allocate input vectors h_A and h_B in host memory

    int size = N * sizeof(float);
    int sb = sizeof(float);

    h_A = (float*)malloc(size);
    //h_B = (float*)malloc(size);
    h_C = (float*)malloc(sb);     // contains the result of dot-product from each block
    
    // Initialize input vectors

    RandomInit(h_A, N);
    //RandomInit(h_B, N);


    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_A, size);
    //cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, sb);

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    int sm = threadsPerBlock*sizeof(float);
    VecDot <<< blocksPerGrid, threadsPerBlock, sm >>>(d_A, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",(2*N-1)/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result of each block in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    //cudaFree(d_B);
    cudaFree(d_C);

    float h_G=0.0;
    //for(int i = 0; i < blocksPerGrid; i++) 
    h_G = h_C[0];
    

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    // to compute the reference solution

    double h_D=0.0;       
    for(int i = 0; i < N; i++) 
      h_D = fmaxf(h_D, h_A[i]);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    //printf("Processing time for CPU: %f (ms) \n",cputime);
    //printf("CPU Gflops: %f\n",(2*N-1)/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    float diff = abs((h_D - h_G));
    printf("|(h_G - h_D)/h_D|=%20.15e\n",diff);
    //printf("h_G =%20.15e\n",h_G);
    //printf("h_D =%20.15e\n",h_D);
	myfile<<threadsPerBlock<<","<<blocksPerGrid<<","<<Outime<<","<<gputime<<","<<Intime <<","<<gputime_tot<<","<<diff<<","<<cputime<<","<< (cputime/(gputime_tot))<<std::endl;
    free(h_A);
    //free(h_B);
    free(h_C);

    cudaDeviceReset();


}


// Allocates an array with random float entries in (-1,1)
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = 2.0*rand()/(float)RAND_MAX - 1.0;
}



// main

int main(void){
	std::ofstream myfile;
	myfile.open("Output.csv");
	myfile<<"threadsPerBlock"<<","<<"blocksPerGrid"<<","<<"Outime"<<","<<"gputime"<<","<<"Intime"<<","<<"gputime_tot"<<","<<"diff"<<","<<"cputime"<<","<<"savetime"<<std::endl;
    int gid;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU ID: ");
    scanf("%d",&gid);
    printf("%d\n", gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Vector Dot Product: A.B\n");
    int N = 81920007;


    // Set the sizes of threads and blocks

    int threadsPerBlocks[6] = {2, 8, 16, 64, 256, 1024};

//    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
//    printf("The number of blocks per grid:%d\n",blocksPerGrid);
 
    int blocksPerGrids[6] = {10, 100, 1000, int(1e3), int(1e4), int(1e5)};

	for(int i = 0; i<6; i++){
		for(int j=0; j<6; j++){
			opt(gid, N, threadsPerBlocks[i], blocksPerGrids[j], myfile);
		}
	}
}

