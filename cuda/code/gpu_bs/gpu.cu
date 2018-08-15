#include <stdlib.h>

#include <stdio.h>

#include <cuda_runtime.h>

#include <sys/time.h>



#define NUM_STEPS    2048

#define BLOCK_SIZE   128

#define MAX_OPTIONS  1024

#define ELEMS_PER_THREAD (NUM_STEPS/BLOCK_SIZE)

/* at T_N = NUM_STEPS, there are (NUM_STEPS + 1) leaves*/



double cpuSecond( ) {

    double sec;

    struct timeval tp;

    gettimeofday(&tp, NULL);

    sec = (double) tp.tv_sec + (double) tp.tv_usec*1e-6;

    return sec;

}



// cumulative standard normal distribution

__device__ __host__ double CND(double d) {



    const double b  =  0.2316419;

    const double a1 =  0.31938153;

    const double a2 = -0.356563782;

    const double a3 =  1.781477937;

    const double a4 = -1.821255978;

    const double a5 =  1.330274429;



    const double rsqrt2pi  = 0.3989422804014327;

    double K = 1.0/(1.0 + b*abs(d));



    double ans;



    ans  = K*(a1 + K*(a2 + K*(a3 + K*(a4 + K*a5) ) ) );

    ans *= rsqrt2pi*exp(-d*d/2.0);

    if (d > 0) {

        ans = 1.0 - ans;

    }

    return ans;

}



__host__ double

BlackScholes(double S0, double X, double r, double v, double T) {



    double sqrtT = sqrt(T);

    double d1    = ( log(S0/X) + (r + 0.5*v*v)*T )/(v*sqrtT);

    double d2    = d1 - v*sqrtT;

    double expRT = exp(-r*T);

    double CE;



    CE = S0*CND(d1) - X*expRT*CND(d2);

    return CE;

}





/* CPU verion of the binomial model */

__host__ double 

binomial(double S, double X, double r, double v, double T, int N, double* V ) {

// S current stock price; X strike price; 

// r, v the risk-free interest rate and volatility;

// T the expiry (unit yr); N the number of time steps in the binomial model.

// V[N+1] is the array to do the iteration;

// In the end, V[0] stores the price of the call option



    double  dt    = T/(double) N;

    double  vdt   = v*sqrt(dt);

    double  u     = exp( vdt );

    double  d     = 1.0/u;

    double  disc  = exp( r * dt );       // discounting factor

    double  discr = 1.0/disc;

    double  pu    = (disc - d)/(u - d);  // risk-neutral/martingale probability

    double  pd    = 1.0 - pu;

    int     i, j;

    double  Si;                          // intermediate stock price in the node



    // initialize the CALL option value on expiry;

    for (i = 0; i <= N; i++) {

        Si   = S * exp( vdt*(2*i - N) );     	// S[0] lowest stock price at expiry. 

        V[i] = (Si - X > 0) ? (Si - X) : 0;     // Call option, use "max(X-Si, 0.0)" for puts  

    }



    // iterate backward of the binomial tree (j the time step).

    for (j = N-1; j >= 0; j--) {

        for (i = 0; i <= j; i++) {

            V[i] =  (pd * V[i] + pu * V[i+1]) * discr;

        }

    }



    return V[0];

}



/* 1D grid, and 1D block; blocksize << NUM_STEPS

   each block  deals with only one option 

   each thread deals with part of the binomial tree */

__global__ void

binomialGPUv5(double* Sptr, double* Xptr, double* Cptr,

              double r, double v, double T) {



    int     tx    = threadIdx.x;

    int     bx    = blockIdx.x;

    double  S     = Sptr[bx];

    double  X     = Xptr[bx];

    double  dt    = T/(double) NUM_STEPS;

    double  vdt   = v*sqrt(dt);

    double  u     = exp(vdt);

    double  d     = 1.0/u;

    double  disc  = exp( r * dt );       // discounting factor

    double  discr = 1.0/disc;

    double  pu    = (disc - d)/(u - d);  // risk-neutral/martingale probability

    double  pd    = 1.0 - pu;

    

    int     i, j, k;

    double  Si;



    double  call_loc[ELEMS_PER_THREAD + 1]; //local array

    __shared__ 

    double  call_bound[BLOCK_SIZE + 1];



    for (i = 0; i < ELEMS_PER_THREAD; i++) {

        k  = tx * ELEMS_PER_THREAD + i;

        Si = S * exp( vdt * (2*k - NUM_STEPS) );

        call_loc[i] =  (Si - X > 0) ? (Si - X) : 0;

    }



    if (tx == BLOCK_SIZE - 1) {

        Si = S * exp( vdt * NUM_STEPS);

        call_bound[BLOCK_SIZE] = (Si - X > 0) ? (Si - X) : 0;

    }

    call_bound[tx] = call_loc[0];

    __syncthreads();

    call_loc[ELEMS_PER_THREAD] = call_bound[tx + 1];

    __syncthreads();



    for (j = 0; j < (BLOCK_SIZE - tx)*ELEMS_PER_THREAD; j++) {



        for (i = 0; i < ELEMS_PER_THREAD; i++) {

            call_loc[i] =  (pd*call_loc[i] + pu*call_loc[i+1])*discr;

        }

        call_bound[tx] = call_loc[0];

        __syncthreads();

        call_loc[ELEMS_PER_THREAD] = call_bound[tx + 1];

        __syncthreads();  // must be there, otherwise, fail

    }



    if (tx == 0) {

        Cptr[bx] = call_loc[0];

    }

}



/* general uniform random number between [low, high] */

double UniRand(double low, double high) {



    double t = (double) rand() / (double) RAND_MAX;

    return (1.0 - t)*low + t*high;

}



/* test the GPU binomial model for European Call Pricing */

int main(int argc, char** argv) {



    int      OPT_N   = MAX_OPTIONS;

    double * Sptr_h;

    double * Sptr_d;

    double * Xptr_h;

    double * Xptr_d;

    double * Cptr_h;

    double * Cptr_d;

    double * Cptr_c;

    double * Cptr_b;  		// from the Black-Scholes model

    int      size = OPT_N*sizeof(double);    

    int      i;

  

    double * Vptr_h; // used by the CPU verions





    double   r = 0.02;  // risk-free rate

    double   v = 0.30;  // volatility

    double   T = 1.00;  // maturity



    cudaError_t error;

    double   tStart;

    double   tStop1, tStop2;



    double   sumDel = 0, sumRef = 0;



    Sptr_h = (double *) malloc(size); // input

    Xptr_h = (double *) malloc(size); // input

    Cptr_h = (double *) malloc(size); // CPU version output

    Cptr_c = (double *) malloc(size); // GPU version output

    Cptr_b = (double *) malloc(size); // CPU Black-Scholes output



    Vptr_h = (double *) malloc((NUM_STEPS+1)*sizeof(double));



    for (i = 0; i < OPT_N; i++) {

        Sptr_h[i] = UniRand(5.0, 30.0);

        Xptr_h[i] = UniRand(1.0, 40.0);

    }



    cudaMalloc( (void **) &Sptr_d, size);

    cudaMalloc( (void **) &Xptr_d, size);

    cudaMalloc( (void **) &Cptr_d, size);



    cudaMemcpy(Sptr_d, Sptr_h, size, cudaMemcpyHostToDevice);

    cudaMemcpy(Xptr_d, Xptr_h, size, cudaMemcpyHostToDevice);



    dim3 dimGrid(MAX_OPTIONS);

    dim3 dimBlock(BLOCK_SIZE);

 

    printf("Depth of the tree NUM_STEPS = %d\n", NUM_STEPS); 

    printf("grid structure: <<<%d, %d>>>\n", dimGrid.x, dimBlock.x);

    printf("Starting the GPU code...\n");



    tStart = cpuSecond();

    cudaDeviceSynchronize();

    binomialGPUv5<<<dimGrid, dimBlock>>>(Sptr_d, Xptr_d, Cptr_d, r, v, T);

    cudaMemcpy(Cptr_c, Cptr_d, size, cudaMemcpyDeviceToHost);   

    error = cudaPeekAtLastError();

    if (error != cudaSuccess) {

        printf("GPU code failed %s\n", cudaGetErrorString(error) ); 

        exit(-1);

    } else {

        tStop1 = cpuSecond() - tStart;

        printf("GPU code finished within %12.6f seconds\n", tStop1);

    }



    tStart = cpuSecond();

    for (i = 0; i < OPT_N; i++) {

        Cptr_h[i] = binomial(Sptr_h[i], Xptr_h[i], r, v, T, NUM_STEPS, Vptr_h);

    }

    tStop2  = cpuSecond() - tStart;

    printf("CPU code finished within %12.6f seconds\n", tStop2);

    printf("Speed up you got %8.2f\n", tStop2/tStop1);



    for (i = 0; i < OPT_N; i++) {

        Cptr_b[i] = BlackScholes(Sptr_h[i], Xptr_h[i], r, v, T);

    }



    printf("Compare the GPU and CPU binary model now...\n");

    for (i = 0; i < OPT_N; i++) {

        if ( abs( Cptr_h[i] - Cptr_c[i] ) > 1e-8 ) {

            printf("%d %8.2f %8.2f %12.6f %12.6f \n", i, Sptr_h[i], Xptr_h[i], Cptr_h[i], Cptr_c[i]);      

            printf("comparion failed\n");

	    exit(-1);

        } 

    }

    printf("passed comparison between GPU and CPU binomial model\n");



    printf("Compare the binary model with Black-Scholes model now...\n");

    for (i = 0; i < OPT_N; i++) {

        sumDel += abs( Cptr_h[i] - Cptr_b[i] ); 

        sumRef += Cptr_h[i];

    }

    if ( sumDel/sumRef > 1e-4 ) {

        printf("Black-Scholes VS Binomial comparison failed\n");

	exit(-1);

    } 

    printf("passed comparison between binomial and Black-Scholes model\n");



    printf("Here are outputs for the first 10 lines\n");

    for (i = 0; i < 10; i++) {

        printf("%8.2f, %8.2f, %10.4f, %10.4f, %10.4f\n", 

              Sptr_h[i], Xptr_h[i], Cptr_c[i], Cptr_h[i], Cptr_b[i]);

    }



    free(Sptr_h); free(Xptr_h); free(Cptr_h); free(Cptr_c); free(Cptr_b); free(Vptr_h);

    cudaFree(Sptr_d); cudaFree(Xptr_d); cudaFree(Cptr_d);

    cudaDeviceReset();



    return(0);

}


