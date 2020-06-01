/*****************************************************************************************
* autoT_A: compute the integrated autocorrelation time to estimate the error of the mean *           
*****************************************************************************************/
/*
   To compile, type the following command.

   g++ -o autoT_A autoT_A.c

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[]) 
{ 
    int i, j, t, N, tmax, count; 
    double *A;                    // observable
    double *A_cor, *tauA;
    double A_ave, C;  
    double tmp1, tmp2, tmp3;
    FILE   *infile, *outfile; 
    //infile = fopen("input.dat","r");
    char *name = argv[1];
		size_t len;
		int read;
		/*
		printf("Enter the input data file name: ");
		//gets(name);                   
		read = getline(&name,&len, stdin);
		*/
		printf("%s\n",name);
		infile = fopen(name,"r");
    
		outfile = fopen("autoT_A.dat","w");
    printf("Enter the number of measurements: ");
    N = 1000;
		//scanf("%d", &N);
    printf("%d\n",N);
    printf("Enter t_max for computing integ. autocorrel. time: ");
    tmax = 1000;
		//scanf("%d", &tmax); 
    printf("%d\n", tmax); 

    int N1=N+1;
    A = (double*) malloc( N1*sizeof(double) );   // array from 0 to N

    int tmax1 = tmax+1;
    A_cor = (double*) malloc( tmax1*sizeof(double) );   // array from 0 to tmax
    tauA = (double*) malloc( tmax1*sizeof(double) );

    for(i=-2;i<N1;i++) {
//      fscanf(infile,"%lf %lf %lf", &tmp1, &A[i], &tmp3);  // read A[i] according to the format of data
      if(i<1)
			{		fscanf(infile,"%*[^\n]\n");  // read A[i] according to the format of data
			}else{
					//fscanf(infile,"%lf %lf", &A[i], &tmp2);  // read A[i] according to the format of data
					fscanf(infile,"%lf %lf", &tmp2, &A[i]);  // read A[i] according to the format of data
					//printf("%lf, %lf\n", A[i], tmp2);  // read A[i] according to the format of data
    	}
		}
    fclose(infile);
          
    A_ave = 0.0;
    double A2_ave = 0.0;
    for(i=1; i<N1; i++) {     // sum from i=1 to N
      A_ave += A[i];
      A2_ave += A[i]*A[i];
    }
    A_ave /= N;  
    A2_ave /= N;
    double err = sqrt( (A2_ave - A_ave*A_ave)/fmax(1,N-1) );

    for(t=0;t<tmax1;t++) {   
      A_cor[t]=0.0;
      count = 0;
      for(i=1; i<N-t+1; i++) {
        A_cor[t] += A[i]*A[i+t];   
        count += 1;
      }
      A_cor[t] = A_cor[t]/count - A_ave*A_ave; 
//      printf("%d %d %e\n",t,count,A_cor[t]);
    }

    int flag = 0;
    double eps = 1.0e-12;
    for(t=1;t<tmax1;t++) {
      tauA[t]=0.5;            
      for(j=1;j<(t+1);j++) {
        C = A_cor[j]/A_cor[0];
        if(C < eps) {   // exit if C < eps 
          flag = 1;
          goto EA;    
        }
        tauA[t] += C;
      }
    } 
    if(flag == 0) {
      printf("t = %d, tau_int = %e (not saturated yet !)\n",t,tauA[t]); 
      fprintf(outfile, "t = %d, tau_int = %e (not saturated yet !)\n",t,tauA[t]); 
    }
EA:
    if(flag == 1) {
      double errMC = sqrt(2.0*tauA[t]); 
      printf("t = %d, tau_int = %e, sqrt(2*tau_int) = %e\n",t,tauA[t],errMC); 
      printf("<A> = %e +- %e\n", A_ave, err*errMC); 
      printf("<A> = %e +- %e (incorrect estimate) \n", A_ave, err); 
      fprintf(outfile,"t = %d, tau_int = %e, sqrt(2*tau_int) = %e\n",t,tauA[t],errMC); 
      fprintf(outfile,"<A> = %e +- %e\n", A_ave, err*errMC); 
      fprintf(outfile,"<A> = %e +- %e (incorrect estimate) \n", A_ave, err); 
    }

    fclose(outfile);
}             

