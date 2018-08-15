#include <stdio.h>
#include <cuda_runtime.h>
__global__ void checkIdx() {
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int tz = threadIdx.z;
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int bz = blockIdx.z;
     printf("threadIdx (%d,%d,%d), gridIdx (%d,%d,%d)\n",
             tx,ty,tz,bx,by,bz);
}
int main(){
    int  nElem = 15;
    dim3 dimBlock(4,1,1);
    dim3 dimGrid( (nElem + dimBlock.x - 1)/dimBlock.x,  1, 1);
    printf("blockdim = (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("griddim  = (%d, %d, %d)\n", dimGrid.x,  dimGrid.y,  dimGrid.z);
    checkIdx<<<dimBlock, dimGrid>>>();
    cudaDeviceReset();
    return 0;
}
