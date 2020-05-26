//  Monte Carlo simulation of Ising model on 2D lattice
//  using Metropolis algorithm
//  using checkerboard (even-odd) update 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <omp.h>

gsl_rng *rng=NULL;    // pointer to gsl_rng random number generator

void exact_2d(double, double, double*, double*);
void rng_MT(float*, int);

double ellf(double phi, double ak);
double rf(double x, double y, double z);
double min(double x, double y, double z);
double max(double x, double y, double z);

int *spin;            // host spin variables
int *d_spin;          // device spin variables
float *h_rng;         // host random numbers
float *d_rng;         // device random numbers 

__constant__ int fw[1000],bw[1000];     // declare constant memory for fw, bw 
int const a = 2;

__global__ void metro_gmem_odd_mgpu(int* spin, float *ranf, const float B, const float T, int* dL_spin,int* dR_spin,int* dB_spin,int* dT_spin)
{
		int Lx = blockDim.x*gridDim.x;
		int Ly = blockDim.y*gridDim.y;
    
		int    x, y, parity;
    int    i, io;
    int    old_spin, new_spin, spins;
    int    k1, k2, k3, k4;
		int 	 t, l, r, b;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(nx,ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
 		 
    int Nx = a*blockDim.x;             // block size before even-odd reduction
    int nx = a*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 

    // next, go over the odd sites 

    io = threadIdx.x + threadIdx.y*blockDim.x;   
    x = (2*io)%Nx;
    y = ((2*io)/Nx)%Nx;
    parity=(x+y+1)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice

    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*nx;
    old_spin = spin[i];
    new_spin = -old_spin;
    k1 = fw[x] + y*nx;     // right
    k2 = x + fw[y]*nx;     // top
    k3 = bw[x] + y*nx;     // left
    k4 = x + bw[y]*nx;     // bottom
		r = spin[k1];
		t = spin[k2];
		l = spin[k3];
		b = spin[k4];
		if(x == 0){ // left == 0
			l = dL_spin[(Lx-1)+y*Lx];
			r = spin[i+1];
		}else if(x == Lx - 1){ // right == O.O.B
			l = spin[i-1];
			r = dR_spin[y*Lx];
		}
		if(y == 0){ // bottom == 0
			b = dB_spin[x+(Ly-1)*Lx];
			t = spin[i+Lx];
		}else if(y == Ly - 1){ // top == O.O.B
			b = spin[i-Lx];
			t = dT_spin[x];
		}
    //spins = spin[k1] + spin[k2] + spin[k3] + spin[k4];
    spins = l + r + b + t;
    //printf("l=%d  r=%d  b=%d  t=%d\n", l, r, b, t);
    //printf("--- s1=%d  s2=%d  s3=%d  s4=%d\n", spin[k1], spin[k2], spin[k3], spin[k4]);
    de = -(new_spin - old_spin)*(spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }

    __syncthreads();

}

__global__ void metro_gmem_even_mgpu(int* spin, float *ranf, const float B, const float T, int* dL_spin,int* dR_spin,int* dB_spin,int* dT_spin)
{
		int Lx = blockDim.x*gridDim.x;
		int Ly = blockDim.y*gridDim.y;
		
    int    x, y, parity;
    int    i, ie;
    int    old_spin, new_spin, spins;
		int    k1, k2, k3, k4;
		int 	 t, l, r, b;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(nx,ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = a*blockDim.x;             // block size before even-odd reduction
    int nx = a*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 

    // first, go over the even sites 

    ie = threadIdx.x + threadIdx.y*blockDim.x;  
    x = (2*ie)%Nx;
    y = ((2*ie)/Nx)%Nx;
    parity=(x+y)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice

    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*nx;
    old_spin = spin[i];
    new_spin = -old_spin;
    k1 = fw[x] + y*nx;     // right
    k2 = x + fw[y]*nx;     // top
    k3 = bw[x] + y*nx;     // left
    k4 = x + bw[y]*nx;     // bottom
		r = spin[k1];
		t = spin[k2];
		l = spin[k3];
		b = spin[k4];
		if(x == 0){ // left == 0
			l = dL_spin[(Lx-1)+y*Lx];
			r = spin[i+1];
		}else if(x == Lx - 1){ // right == O.O.B
			l = spin[i-1];
			r = dR_spin[y*Lx];
		}
		if(y == 0){ // bottom == 0
			b = dB_spin[x+(Ly-1)*Lx];
			t = spin[i+Lx];
		}else if(y == Ly - 1){ // top == O.O.B
			b = spin[i-Lx];
			t = dT_spin[x];
		}
    //spins = spin[k1] + spin[k2] + spin[k3] + spin[k4];
    spins = l + r + b + t;
    de = -(new_spin - old_spin)*(spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }
    
    __syncthreads();
 
}   




int main(void) {
  int nx,ny; 		// # of sites in x and y directions respectively
  int ns; 		// ns = nx*ny, total # of sites
  int *ffw;      	// forward index
  int *bbw; 	        // backward index
  int nt; 		// # of sweeps for thermalization
  int nm; 		// # of measurements
  int im; 		// interval between successive measurements
  int nd; 		// # of sweeps between displaying results
  int nb; 		// # of sweeps before saving spin configurations
  int sweeps; 		// total # of sweeps at each temperature
  int k1, k2;           // right, top
  int istart; 		// istart = (0: cold start/1: hot start)
  double T; 		// temperature
  double B; 		// external magnetic field
  double energy; 	// total energy of the system
  double mag; 		// total magnetization of the system
  double te; 		// accumulator for energy
  double tm; 		// accumulator for mag
  double count; 	// counter for # of measurements
  double M; 		// magnetization per site, < M >
  double E; 		// energy per site, < E >
  double E_ex; 		// exact solution of < E >
  double M_ex; 		// exact solution of < M >
	int cpu_thread_id = 0;

	float** md_rng;
	int** md_spin;

  int gid;              // GPU_ID
  float gputime;
  float flops;
	
  int Dev[2] = {0, 1};
  int NGPU = 2;
	int NGx = 1;
	int NGy = 2;

	
	printf("MGPU! \n");
  
  printf("Enter the GPU ID (0/1): ");
  scanf("%d",&gid);
  printf("%d\n",gid);

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;
  err = cudaSetDevice(gid);
  if(err != cudaSuccess) {
    printf("!!! Cannot select GPU with device ID = %d\n", gid);
    exit(1);
  }
  printf("Select GPU with device ID = %d\n", gid);
  cudaSetDevice(gid);

  printf("Ising Model on 2D Square Lattice with p.b.c.\n");
  printf("============================================\n");
  printf("Initialize the RNG\n");
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  printf("Enter the seed:\n");
  long seed;
  scanf("%ld",&seed);
  printf("%ld\n",seed); 
  gsl_rng_set(rng,seed);
  printf("The RNG has been initialized\n");
  printf("Enter the number of sites in each dimension (<= 1000)\n");
  scanf("%d",&nx);
  printf("%d\n",nx);
  ny=nx;
  ns=nx*ny;
  ffw = (int*)malloc(nx*sizeof(int));
  bbw = (int*)malloc(nx*sizeof(int));
  for(int i=0; i<nx; i++) {
    ffw[i]=(i+1)%nx;
    bbw[i]=(i-1+nx)%nx;
  }
	
	int Lx = nx/NGx;
	int Ly = ny/NGy;

  cudaMemcpyToSymbol(fw, ffw, nx*sizeof(int));  // copy to constant memory
  cudaMemcpyToSymbol(bw, bbw, nx*sizeof(int));

  spin = (int*)malloc(ns*sizeof(int));          // host spin variables
  h_rng = (float*)malloc(ns*sizeof(float));     // host random numbers

  printf("Enter the # of sweeps for thermalization\n");
  scanf("%d",&nt);
  printf("%d\n",nt);
  printf("Enter the # of measurements\n");
  scanf("%d",&nm);
  printf("%d\n",nm);
  printf("Enter the interval between successive measurements\n");
  scanf("%d",&im);
  printf("%d\n",im);
  printf("Enter the display interval\n");
  scanf("%d",&nd);
  printf("%d\n",nd);
  printf("Enter the interval for saving spin configuration\n");
  scanf("%d",&nb);
  printf("%d\n",nb);
  printf("Enter the temperature (in units of J/k)\n");
  scanf("%lf",&T);
  printf("%lf\n",T);
  printf("Enter the external magnetization\n");
  scanf("%lf",&B);
  printf("%lf\n",B);
  printf("Initialize spins configurations :\n");
  printf(" 0: cold start \n");
  printf(" 1: hot start \n");
  scanf("%d",&istart);
  printf("%d\n",istart);
 
  // Set the number of threads (tx,ty) per block

  int tx,ty;
  printf("Enter the number of threads (tx,ty) per block: ");
  printf("For even/odd updating, tx=ty/2 is assumed: ");
  scanf("%d %d",&tx, &ty);
  printf("%d %d\n",tx, ty);
  if(2*tx != ty) exit(0);
  if(tx*ty > 1024) {
    printf("The number of threads per block must be less than 1024 ! \n");
    exit(0);
  }
  dim3 threads(tx,ty);

  // The total number of threads in the grid is equal to (nx/2)*ny = ns/2 

  int bx = nx/tx/2;
  if(bx*tx*2 != nx) {
    printf("The block size in x is incorrect\n");
    exit(0);
  }
  int by = ny/ty;
  if(by*ty != ny) {
    printf("The block size in y is incorrect\n");
    exit(0);
  }
  if((bx > 65535)||(by > 65535)) {
    printf("The grid size exceeds the limit ! \n");
    exit(0);
  }
  dim3 blocks(bx/NGx,by/NGy);
  printf("The dimension of the grid is (%d, %d)\n",bx,by);

  if(istart == 0) {
    for(int j=0; j<ns; j++) {       // cold start
      spin[j] = 1;
    }
  }
  else {
    for(int j=0; j<ns; j++) {     // hot start
      if(gsl_rng_uniform(rng) > 0.5) { 
        spin[j] = 1;
      }
      else {
        spin[j] = -1;
      }
    }
  }

  FILE *output;            
  output = fopen("ising2d_2gpu_gmem.dat","w");
  FILE *output3;
  output3 = fopen("spin_2gpu_gmem.dat","w");   

  // Allocate vectors in device memory

  cudaMalloc((void**)&d_spin, ns*sizeof(int));         // device spin variables
  cudaMalloc((void**)&d_rng, ns*sizeof(float));        // device random numbers

  // Copy vectors from host memory to device memory

  cudaMemcpy(d_spin, spin, ns*sizeof(int), cudaMemcpyHostToDevice);

  if(B == 0.0) {
    exact_2d(T,B,&E_ex,&M_ex);
    fprintf(output,"T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    fprintf(output,"T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }
  fprintf(output,"     E           M        \n");
  fprintf(output,"--------------------------\n");

  printf("Thermalizing\n");
  printf("sweeps   < E >     < M >\n");
  printf("---------------------------------\n");
  fflush(stdout);

  te=0.0;                          //  initialize the accumulators
  tm=0.0;
  count=0.0;
  sweeps=nt+nm*im;                 //  total # of sweeps

  
	/* 
		Allocate working space for GPUs.
  	spin = (int*)malloc(ns*sizeof(int));          // host spin variables
  	h_rng = (float*)malloc(ns*sizeof(float));     // host random numbers
	*/


	printf("\n* Allocate working space for GPUs ....\n");
	//?sm = tx*ty*sizeof(float);	// size of the shared memory in each block

	md_rng = (float **)malloc(NGPU*sizeof(float *));
	md_spin = (int **)malloc(NGPU*sizeof(int *));

	omp_set_num_threads(NGPU);
	#pragma omp parallel private(cpu_thread_id)
	{
		int cpuid_x, cpuid_y;
		cpu_thread_id = omp_get_thread_num();
		cpuid_x       = cpu_thread_id % NGx;
		cpuid_y       = cpu_thread_id / NGx;
		cudaSetDevice(Dev[cpu_thread_id]);

		int cpuid_r = ((cpuid_x+1)%NGx) + cpuid_y*NGx;         // GPU on the right
		cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
		int cpuid_l = ((cpuid_x+NGx-1)%NGx) + cpuid_y*NGx;     // GPU on the left
		cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
		int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;         // GPU on the top
		cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
		int cpuid_b = cpuid_x + ((cpuid_y+NGy-1)%NGy)*NGx;     // GPU on the bottom
		cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);


		// Allocate vectors in device memory
		cudaMalloc((void**)&md_rng[cpu_thread_id], ns*sizeof(float)/NGPU);
		cudaMalloc((void**)&md_spin[cpu_thread_id], ns*sizeof(int)/NGPU);

		// Copy vectors from the host memory to the device memory

		for (int i=0; i < Ly; i++) {
			int *h, *d;
			h = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*nx;
			d = md_spin[cpu_thread_id] + i*Lx;
			cudaMemcpy(d, h, Lx*sizeof(int), cudaMemcpyHostToDevice);
		}

		#pragma omp barrier

	} // OpenMP
  


  // create the timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //start the timer
  cudaEventRecord(start,0);

  for(int swp=0; swp<nt; swp++) {      // thermalization
		rng_MT(h_rng, ns);                                  // generate ns random numbers 

		#pragma omp parallel private(cpu_thread_id)
		{
			//parallel
			int cpuid_x, cpuid_y;
			cpu_thread_id = omp_get_thread_num();
			cpuid_x       = cpu_thread_id % NGx;
			cpuid_y       = cpu_thread_id / NGx;
			cudaSetDevice(Dev[cpu_thread_id]);

			//float **d_spin, **d_new;
			int *dL_spin, *dR_spin, *dT_spin, *dB_spin;
			dL_spin = (cpuid_x == 0)     ? md_spin[NGx-1+cpuid_y*NGx] : md_spin[cpuid_x-1+cpuid_y*NGx];
			dR_spin = (cpuid_x == NGx-1) ? md_spin[0+cpuid_y*NGx] : md_spin[cpuid_x+1+cpuid_y*NGx];
			dB_spin = (cpuid_y == 0    ) ? md_spin[cpuid_x+(NGy-1)*NGx] : md_spin[cpuid_x+(cpuid_y-1)*NGx];
			dT_spin = (cpuid_y == NGy-1) ? md_spin[cpuid_x+(0)*NGx] : md_spin[cpuid_x+(cpuid_y+1)*NGx];
			//
			
			for (int i=0; i < Ly; i++) {
				float *h, *d;
				h = h_rng + cpuid_x*Lx + (cpuid_y*Ly+i)*nx;
				d = md_rng[cpu_thread_id] + i*Lx;
				cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
			}

			//rng_MT(h_rng, ns);                                  // generate ns random numbers 
			//cudaMemcpy(d_rng, h_rng, ns*sizeof(float), cudaMemcpyHostToDevice);
			metro_gmem_even_mgpu<<<blocks,threads>>>(md_spin[cpu_thread_id], md_rng[cpu_thread_id], B, T, dL_spin, dR_spin, dB_spin, dT_spin);    // updating with Metropolis algorithm
			metro_gmem_odd_mgpu<<<blocks,threads>>>(md_spin[cpu_thread_id], md_rng[cpu_thread_id], B, T, dL_spin, dR_spin, dB_spin, dT_spin);     // updating with Metropolis algorithm
  
		}
	}


	
	//blocks(bx/NGx,by/NGy);
  
	for(int swp=nt; swp<sweeps; swp++) {
		rng_MT(h_rng, ns);                                  // generate ns random numbers 

		#pragma omp parallel private(cpu_thread_id)
		{
			//parallel
			int cpuid_x, cpuid_y;
			cpu_thread_id = omp_get_thread_num();
			cpuid_x       = cpu_thread_id % NGx;
			cpuid_y       = cpu_thread_id / NGx;
			cudaSetDevice(Dev[cpu_thread_id]);

			//float **d_spin, **d_new;
			int *dL_spin, *dR_spin, *dT_spin, *dB_spin;
			dL_spin = (cpuid_x == 0)     ? md_spin[NGx-1+cpuid_y*NGx] : md_spin[cpuid_x-1+cpuid_y*NGx];
			dR_spin = (cpuid_x == NGx-1) ? md_spin[0+cpuid_y*NGx] : md_spin[cpuid_x+1+cpuid_y*NGx];
			dB_spin = (cpuid_y == 0    ) ? md_spin[cpuid_x+(NGy-1)*NGx] : md_spin[cpuid_x+(cpuid_y-1)*NGx];
			dT_spin = (cpuid_y == NGy-1) ? md_spin[cpuid_x+(0)*NGx] : md_spin[cpuid_x+(cpuid_y+1)*NGx];
			//dL_spin = (cpuid_x == 0)     ? NULL : md_spin[cpuid_x-1+cpuid_y*NGx];
			//dR_spin = (cpuid_x == NGx-1) ? NULL : md_spin[cpuid_x+1+cpuid_y*NGx];
			//dB_spin = (cpuid_y == 0    ) ? NULL : md_spin[cpuid_x+(cpuid_y-1)*NGx];
			//dT_spin = (cpuid_y == NGy-1) ? NULL : md_spin[cpuid_x+(cpuid_y+1)*NGx];
			//
			
			for (int i=0; i < Ly; i++) {
				float *h, *d;
				h = h_rng + cpuid_x*Lx + (cpuid_y*Ly+i)*nx;
				d = md_rng[cpu_thread_id] + i*Lx;
				cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
			}

			//rng_MT(h_rng, ns);                                  // generate ns random numbers 
			//cudaMemcpy(d_rng, h_rng, ns*sizeof(float), cudaMemcpyHostToDevice);
			metro_gmem_even_mgpu<<<blocks,threads>>>(md_spin[cpu_thread_id], md_rng[cpu_thread_id], B, T, dL_spin, dR_spin, dB_spin, dT_spin);    // updating with Metropolis algorithm
			metro_gmem_odd_mgpu<<<blocks,threads>>>(md_spin[cpu_thread_id], md_rng[cpu_thread_id], B, T, dL_spin, dR_spin, dB_spin, dT_spin);     // updating with Metropolis algorithm
  
		}
		/*
		rng_MT(h_rng, ns);                                  // generate ns random numbers 
		cudaMemcpy(d_rng, h_rng, ns*sizeof(float), cudaMemcpyHostToDevice);
		metro_gmem_even<<<blocks,threads>>>(d_spin, d_rng, B, T);
		metro_gmem_odd<<<blocks,threads>>>(d_spin, d_rng, B, T);
		*/
	int k; 
	if(swp%im == 0) {
		
		// copy to host
		#pragma omp parallel private(cpu_thread_id)
		{
			int cpuid_x, cpuid_y;
			cpu_thread_id = omp_get_thread_num();
			cpuid_x       = cpu_thread_id % NGx;
			cpuid_y       = cpu_thread_id / NGx;
			cudaSetDevice(Dev[cpu_thread_id]);

			int* d_new = md_spin[cpu_thread_id];
			for (int i=0; i < Ly; i++) {
				int *g, *d;
				g = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*nx;
				d = d_new + i*Lx;
				cudaMemcpy(g, d, Lx*sizeof(int), cudaMemcpyDeviceToHost);
			}
			cudaFree(md_rng[cpu_thread_id]);
			cudaFree(md_spin[cpu_thread_id]);
		} // OpenMP

	  //cudaMemcpy(spin, d_spin, ns*sizeof(int), cudaMemcpyDeviceToHost);
	  mag=0.0;
	  energy=0.0;
	  for(int j=0; j<ny; j++) {
		for(int i=0; i<nx; i++) {
		  k = i + j*nx;
		  k1 = ffw[i] + j*nx;
		  k2 = i + ffw[j]*nx;
		  mag = mag + spin[k]; // total magnetization;
		  energy = energy - spin[k]*(spin[k1] + spin[k2]);  // total bond energy;
		}
	  }
	  energy = energy - B*mag;
	  te = te + energy;
	  tm = tm + mag;
	  count = count + 1.0;
	  fprintf(output, "%.5e  %.5e\n", energy/(double)ns, mag/(double)ns);  // save the raw data 
	}
	if(swp%nd == 0) {
	  E = te/(count*(double)(ns));
	  M = tm/(count*(double)(ns));
	  printf("%d  %.5e  %.5e\n", swp, E, M);
	}
	if(swp%nb == 0) {
	  cudaMemcpy(spin, d_spin, ns*sizeof(int), cudaMemcpyDeviceToHost);
	  fprintf(output3,"swp = %d, spin configuration:\n",swp);
	  for(int j=nx-1;j>-1;j--) {
		for(int i=0; i<nx; i++) {
		  fprintf(output3,"%d ",spin[i+j*nx]);
		}
		fprintf(output3,"\n");
	  }
	  fprintf(output3,"\n");
	}
  }
  fclose(output);      
  fclose(output3);
  printf("---------------------------------\n");
  if(B == 0.0) {
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }
	


  

  // stop the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&gputime, start, stop);
  printf("Processing time for GPU: %f (ms) \n",gputime);
  flops = 7.0*nx*nx*sweeps;
  printf("GPU Gflops: %lf\n",flops/(1000000.0*gputime));

  // destroy the timer
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  gsl_rng_free(rng);
  cudaFree(d_spin);
  cudaFree(d_rng);

  free(spin);
  free(h_rng);

  return 0;
}
          
          
// Exact solution of 2d Ising model on the infinite lattice

void exact_2d(double T, double B, double *E, double *M)
{
  double x, y;
  double z, Tc, K, K1;
  const double pi = acos(-1.0);
    
  K = 2.0/T;
  if(B == 0.0) {
    Tc = -2.0/log(sqrt(2.0) - 1.0); // critical temperature;
    if(T > Tc) {
      *M = 0.0;
    }
    else if(T < Tc) {
      z = exp(-K);
      *M = pow(1.0 + z*z,0.25)*pow(1.0 - 6.0*z*z + pow(z,4),0.125)/sqrt(1.0 - z*z);
    }
    x = 0.5*pi;
    y = 2.0*sinh(K)/pow(cosh(K),2);
    K1 = ellf(x, y);
    *E = -1.0/tanh(K)*(1. + 2.0/pi*K1*(2.0*pow(tanh(K),2) - 1.0));
  }
  else
    printf("Exact solution is only known for B=0 !\n");
    
  return;
}


/*******
* ellf *      Elliptic integral of the 1st kind 
*******/

double ellf(double phi, double ak)
{
  double ellf;
  double s;

  s=sin(phi);
  ellf=s*rf(pow(cos(phi),2),(1.0-s*ak)*(1.0+s*ak),1.0);

  return ellf;
}

double rf(double x, double y, double z)
{
  double rf,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4;
  ERRTOL=0.08; 
  TINY=1.5e-38; 
  BIG=3.0e37; 
  THIRD=1.0/3.0;
  C1=1.0/24.0; 
  C2=0.1; 
  C3=3.0/44.0; 
  C4=1.0/14.0;
  double alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;
    
  if(min(x,y,z) < 0 || min(x+y,x+z,y+z) < TINY || max(x,y,z) > BIG) {
    printf("invalid arguments in rf\n");
    exit(1);
  }

  xt=x;
  yt=y;
  zt=z;

  do {
    sqrtx=sqrt(xt);
    sqrty=sqrt(yt);
    sqrtz=sqrt(zt);
    alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
    xt=0.25*(xt+alamb);
    yt=0.25*(yt+alamb);
    zt=0.25*(zt+alamb);
    ave=THIRD*(xt+yt+zt);
    delx=(ave-xt)/ave;
    dely=(ave-yt)/ave;
    delz=(ave-zt)/ave;
  } 
  while (max(abs(delx),abs(dely),abs(delz)) > ERRTOL);

  e2=delx*dely-pow(delz,2);
  e3=delx*dely*delz;
  rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
    
  return rf;
}

double min(double x, double y, double z)
{
  double m;

  m = (x < y) ? x : y;
  m = (m < z) ? m : z;

  return m;
}

double max(double x, double y, double z)
{
  double m;

  m = (x > y) ? x : y;
  m = (m > z) ? m : z;

  return m;
}

void rng_MT(float* data, int n)   // RNG with uniform distribution in (0,1)
{
    for(int i = 0; i < n; i++)
      data[i] = (float) gsl_rng_uniform(rng); 
}

