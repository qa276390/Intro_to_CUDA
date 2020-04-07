// Solve the Laplace equation on a 2D lattice with boundary conditions.
// (using texture memory)
//
// compile with the following command:
//
// (for GTX970)
// nvcc -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 -o laplace laplace.cu
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o laplace laplace.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

// field variables
float* h_new;   // host field vectors
float* h_old;   
float* h_C;     // sum of diff*diff of each block
float* g_new;   // device solution back to the host 
float* d_new;   // device field vectors
float* d_old;  
float* d_C;     // sum of diff*diff of each block 

int     MAX=1000000;      // maximum iterations
double  eps=1.0e-10;      // stopping criterion


__align__(8) texture<float>  texOld;   // declare the texture
__align__(8) texture<float>  texNew;


__global__ void laplacian(float* phi_old, float* phi_new, float* C, bool flag)
{
    extern __shared__ float cache[];     
    float  t, l, c, r, b, f, k;     // top, left, center, right, bottom
    float  diff; 
    int    site, ym1, xm1, xp1, yp1, zm1, zp1;    

    int Nx = blockDim.x*gridDim.x;
    int Ny = blockDim.y*gridDim.y;
    int Nz = blockDim.z*gridDim.z;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    int cacheIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y*threadIdx.z);  

    site = x + Nx * (y + Ny * z);

    if((x == 0) || (x == Nx-1) || (y == 0) || (y == Ny-1) || (z == 0) || (z==Nz-1) ) {  
      diff = 0.0; 
    }
    else {
      	xm1 = site - 1;    // x-1
      	xp1 = site + 1;    // x+1
		ym1 = x+Nx*((y-1)+Ny*z);   // y-1
		yp1 = x+Nx*((y+1)+Ny*z);   // y+1
		zm1 = x+Nx*(y+Ny*(z-1));   // z-1
		zp1 = x+Nx*(y+Ny*(z+1));   // z+1
	if(flag) {
        b = tex1Dfetch(texOld, ym1);      // read d_old via texOld
        l = tex1Dfetch(texOld, xm1);
        c = tex1Dfetch(texOld, site);
        r = tex1Dfetch(texOld, xp1);
        t = tex1Dfetch(texOld, yp1);
        f = tex1Dfetch(texOld, zp1);
        k = tex1Dfetch(texOld, zm1);
		
        phi_new[site] = (b+l+r+t+f+k)/6.0;
        diff = phi_new[site]-c;
      }
      else {
        b = tex1Dfetch(texNew, ym1);     // read d_new via texNew
        l = tex1Dfetch(texNew, xm1);
        c = tex1Dfetch(texNew, site);
        r = tex1Dfetch(texNew, xp1);
        t = tex1Dfetch(texNew, yp1);
        f = tex1Dfetch(texNew, zp1);
        k = tex1Dfetch(texNew, zm1);
        phi_old[site] = (b+l+r+t+f+k)/6.0;
        diff = phi_old[site]-c;
      }
    }

    // each thread saves its error estimate to the shared memory

    cache[cacheIndex]=diff*diff;  
    __syncthreads();

    // parallel reduction in each block 

    int ib = blockDim.x*blockDim.y*blockDim.z/2;  
    while (ib != 0) {  
      if(cacheIndex < ib)  
        cache[cacheIndex] += cache[cacheIndex + ib];
      __syncthreads();
      ib /=2;  
    } 

    // save the partial sum of each block to C

    int blockIndex = blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y*blockIdx.z);
    if(cacheIndex == 0)  C[blockIndex] = cache[0];    
}


int op(int gid, int CPU, int Nx, int Ny, int Nz, int tx, int ty, int tz, std::ofstream& myfile)
{
	/*
    int iter;
    volatile bool flag;   // to toggle between *_new and *_old  
    float cputime;
    float gputime;
    float gputime_tot;
    double flops;
    */double error;



    dim3 threads(tx,ty, tz); 
    
    // The total number of threads in the grid is equal to the total number of lattice sites
    
    int bx = Nx/tx;
    if(bx*tx != Nx) {
      printf("The blocksize in x is incorrect\n"); 
      exit(0);
    }
    int by = Ny/ty;
    if(by*ty != Ny) {
      printf("The blocksize in y is incorrect\n"); 
      exit(0);
    }
    int bz = Nz/tz;
    if(bz*tz != Nz) {
      printf("The blocksize in z is incorrect\n"); 
      exit(0);
    }
    if((bx > 65535)||(by > 65535)||(bz > 65535)) {
      printf("The grid size exceeds the limit ! \n");
      exit(0);
    }
    dim3 blocks(bx,by,bz);
    printf("The dimension of the grid is (%d, %d, %d)\n",bx,by,bz); 
   
    // Allocate field vector h_phi in host memory

    int N = Nx*Ny*Nz;
    int size = N*sizeof(float);
    int sb = bx*by*bz*sizeof(float);
    h_old = (float*)malloc(size);
    h_new = (float*)malloc(size);
    g_new = (float*)malloc(size);
    h_C = (float*)malloc(sb);
   
    memset(h_old, 0, size);    
    memset(h_new, 0, size);

    // Initialize the field vector with boundary conditions

    for(int x=0; x<Nx; x++) {
		for(int y=0;y<Ny;y++){
      		h_new[x+Nx*(y+Ny*(Nz-1))]=1.0;  
      		h_old[x+Nx*(y+Ny*(Nz-1))]=1.0;  
    	}
	}

    FILE *out1;		        // save initial configuration in phi_initial_3D.dat 
    out1 = fopen("phi_initial_3D.dat","w");

    fprintf(out1, "Inital field configuration:\n");
	for(int k=Nz-1;k>=0;k--){
		for(int j=Ny-1;j>=0;j--) {
		  	for(int i=0; i<Nx; i++) {
				fprintf(out1,"%.2e ",h_new[i+Nx*(j+Ny*k)]);
		  	}
		  	fprintf(out1,"\n");
    	}
		fprintf(out1,"\n");
	}
    fclose(out1);

//   printf("\n");
//    printf("Inital field configuration:\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",h_new[i+j*Nx]);
//      }
//      printf("\n");
//    }
//    printf("\n");

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_new, size);
    cudaMalloc((void**)&d_old, size);
    cudaMalloc((void**)&d_C, sb);

    cudaBindTexture(NULL, texOld, d_old, size);   // bind the texture
    cudaBindTexture(NULL, texNew, d_new, size);

    // Copy vectors from host memory to device memory

    cudaMemcpy(d_new, h_new, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old, h_old, size, cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    error = 10*eps;      // any value bigger than eps is OK
    int iter = 0;        // counter for iterations
    double diff; 

    volatile bool flag = true;     

    int sm = tx*ty*tz*sizeof(float);   // size of the shared memory in each block

    while ( (error > eps) && (iter < MAX) ) {

      laplacian<<<blocks,threads,sm>>>(d_old, d_new, d_C, flag);
      cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
      error = 0.0;
      for(int i=0; i<bx*by*bz; i++) {
        error = error + h_C[i];
      }
      error = sqrt(error);

//      printf("error = %.15e\n",error);
//      printf("iteration = %d\n",iter);

      iter++;
      flag = !flag;
      
    }
     
    printf("error (GPU) = %.15e\n",error);
    printf("total iterations (GPU) = %d\n",iter);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    double flops;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    flops = 7.0*(Nx-2)*(Ny-2)*(Nz-2)*iter;
    printf("GPU Gflops: %f\n",flops/(1000000.0*gputime));
    
    // Copy result from device memory to host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(g_new, d_new, size, cudaMemcpyDeviceToHost);

    cudaFree(d_new);
    cudaFree(d_old);
    cudaFree(d_C);

    cudaUnbindTexture(texOld);
    cudaUnbindTexture(texNew);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);
    fflush(stdout);

    printf("\n");
//    printf("Final field configuration (GPU):\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",g_new[i+j*Nx]);
//      }
//      printf("\n");
//    }
//    printf("\n");

    FILE *outg;                 // save GPU solution in phi_GPU_Tex_3D.dat 
    outg = fopen("phi_GPU_Tex_3D.dat","w");

    fprintf(outg, "GPU (using texture) field configuration:\n");
	for(int k=Nz-1;k>=0;k--){
		for(int j=Ny-1;j>=0;j--) {
		  	for(int i=0; i<Nx; i++) {
				fprintf(outg,"%.2e ",g_new[i+Nx*(j+Ny*k)]);
			}
		  	fprintf(outg,"\n");
		}
		fprintf(outg,"\n");
	}
    fclose(outg);

    // start the timer
    cudaEventRecord(start,0);

    if(CPU==0) {
      free(h_new);
      free(h_old);
      free(g_new);
      free(h_C);
      cudaDeviceReset();
    } 
 
    // to compute the reference solution

    error = 10*eps;      // any value bigger than eps 
    iter = 0;            // counter for iterations
    flag = true;     

    float t, l, r, b, f, k;    // top, left, right, bottom
    int site, ym1, xm1, xp1, yp1, zm1, zp1;

    while ( (error > eps) && (iter < MAX) ) {
      if(flag) {
        error = 0.0;
		for(int z=0; z<Nz; z++) {
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1 || z==0 || z==Nz-1) {   
          }
          else {
            site = x+Nx*(y+Ny*z);
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = x+Nx*((y-1)+Ny*z);   // y-1
            yp1 = x+Nx*((y+1)+Ny*z);   // y+1
            zm1 = x+Nx*(y+Ny*(z-1));   // z-1
            zp1 = x+Nx*(y+Ny*(z+1));   // z+1
            b = h_old[ym1]; 
            l = h_old[xm1];
			k = h_old[zm1];
            r = h_old[xp1]; 
            t = h_old[yp1];
			f = h_old[zp1];
            h_new[site] = (b+l+r+t+f+k)/6.0;
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          }
        } 
        }
		}
      }
      else {
        error = 0.0;
		for(int z=0; z<Nz; z++) {
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1 || z==0 || z==Nz-1) {   
          }
          else {
            site = x+Nx*(y+Ny*z);
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = x+Nx*((y-1)+Ny*z);   // y-1
            yp1 = x+Nx*((y+1)+Ny*z);   // y+1
            zm1 = x+Nx*(y+Ny*(z-1));   // z-1
            zp1 = x+Nx*(y+Ny*(z+1));   // z+1
            b = h_new[ym1]; 
            l = h_new[xm1];
			k = h_new[zm1];
            r = h_new[xp1]; 
            t = h_new[yp1];
			f = h_new[zp1];
            h_old[site] = (b+l+r+t+f+k)/6.0;
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          }
        } 
        }
		}
      }
      flag = !flag;
      iter++;
      error = sqrt(error);

//      printf("error = %.15e\n",error);
//      printf("iteration = %d\n",iter);

    }   // exit if error < eps
    
    printf("error (CPU) = %.15e\n",error);
    printf("total iterations (CPU) = %d\n",iter);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    flops = 7.0*(Nx-2)*(Ny-2)*(Nz-2)*iter;
    printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));
    fflush(stdout);
	myfile<<Nx<<"x"<<Ny<<","<<tx<<"x"<<ty<<","<<gputime<<","<<gputime_tot<<","<<error<<","<<cputime<<","<<(cputime/gputime_tot)<<std::endl;

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    FILE *outc;               // save CPU solution in phi_CPU_3D.dat 
    outc = fopen("phi_CPU_3D.dat","w");

    fprintf(outc,"CPU field configuration:\n");
    for(int z=Nz-1;z>-1;z--) {
    	for(int j=Ny-1;j>-1;j--) {
      		for(int i=0; i<Nx; i++) {
        		fprintf(outc,"%.2e ",h_new[i+Nx*(j+Ny*z)]);
      		}
      		fprintf(outc,"\n");
    	}
      	fprintf(outc,"\n");
	}
    fclose(outc);

//    printf("\n");
//    printf("Final field configuration (CPU):\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",h_new[i+j*Nx]);
//      }
//      printf("\n");
//    }

    free(h_new);
    free(h_old);
    free(h_C);
    free(g_new);

    cudaDeviceReset();
}

int main(void)
{

	int gid;              // GPU_ID

	std::ofstream myfile;
	myfile.open("OutputTex3D.csv");
	myfile<<"lattice_sizes"<<","<<"block sizes"<<","<<"gputime"<<","<<"gputime_tot"<<","<<"diff"<<","<<"cputime"<<","<<"savetime"<<std::endl;

	printf("Enter the GPU ID (0/1): ");
	scanf("%d",&gid);
	printf("%d\n",gid);

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	err = cudaSetDevice(gid);
	if (err != cudaSuccess) {
		printf("!!! Cannot select GPU with device ID = %d\n", gid);
		exit(1);
	}
	printf("Select GPU with device ID = %d\n", gid);

	cudaSetDevice(gid);

	printf("Solve Laplace equation on a 2D lattice with boundary conditions\n");

	int CPU;    
	printf("To compute the solution vector with CPU/GPU/both (0/1/2) ? ");
	scanf("%d",&CPU);
	printf("%d\n",CPU);
	fflush(stdout);

	int Ns[4] = {32, 64, 128, 256};
	//int Ns[2] = {4, 16};
	int ts[4] = {4, 8, 16, 32};
	//int ts[2] = {2, 4};
	for(int i=0; i<4; i++)
	{

		for(int j=0; j<4; j++)
		{
			printf("%d %d\n",i, j);
			op(gid, CPU, Ns[i], Ns[i], Ns[i], ts[j], ts[j], ts[j], myfile);
		}
	}



}
