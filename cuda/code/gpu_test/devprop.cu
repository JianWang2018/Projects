#include <stdio.h>
#include <cuda_runtime.h>
int main(int argc, char ** argv){
    cudaError_t error;
    printf("%s running...\n", argv[0]);
    int   devCount;
    cudaGetDeviceCount(&devCount);
    printf("number of devices: %d\n", devCount);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("maxThreadsPerBlock = %d\n", devProp.maxThreadsPerBlock);
    printf("max block dimension (%d, %d, %d)\n", devProp.maxThreadsDim[0],
    devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("max grid dimension (%d, %d, %d)\n", devProp.maxGridSize[0],
        devProp.maxGridSize[1], devProp.maxGridSize[2]);
    return 0;
}
