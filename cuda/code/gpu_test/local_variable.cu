#include <stdio.h>
#include <cuda_runtime.h>
__global__ void kernel() {
    double  a = 2.71828;    //register variables, automatic
    double  c[100];         //local variable, automatic
    __shared__ double b;    //shared variable
    int  tx  = threadIdx.x; //register variable
    if (tx == 0) {
        b = 3.1415926f;
    }
    //__syncthreads();        // run with/without this line
    printf("id = %d, a=%7.5f, b=%9.7f\n", tx, a, b);
}
int main() {
    kernel<<<1,8>>>();
    cudaDeviceReset();
    return 0;
}
