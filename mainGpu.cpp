#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define NX 800
#define NY 400
#define SIZE NX*NY*3*sizeof(int)

__global__ void generate(int *A){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float r = float(i) / float(NX);
    float g = float(j) / float(NY);        
    float b = 0.2;
    int ir = int(255.99*r);
    int ig = int(255.99*g);
    int ib = int(255.99*b);
    A[(i*NY + j)*3] = ir;
    A[(i*NY + j)*3 + 1] = ig;
    A[(i*NY + j)*3 + 2] = ib;
}

int main() {

    dim3 dimGrid(ceil(NX/(float)16), ceil(NY/(float)16));
    dim3 dimBlock(16, 16);

    // dim3 numBlocks(ceil(, ceil(NY / threadsPerBlock.y));
    int *cpuA;
    int *gpuA;
    cpuA = (int *)malloc(SIZE);
    cudaMalloc((void **)&gpuA,SIZE);
    generate<<<dimGrid, dimBlock>>>(gpuA);
    cudaMemcpy(cpuA, gpuA, SIZE, cudaMemcpyDeviceToHost);
    cudaFree(gpuA);

    std::cout << "P3\n" << NX << " " << NY << "\n255\n";
    for (int j = NY-1; j >= 0; j--){
        for(int i = 0; i < NX; i++){
            std::cout << cpuA[(i*NY + j)*3] << " " << cpuA[(i*NY + j)*3 + 1] << " " << cpuA[(i*NY + j)*3 + 2] << "\n";
        }
    }
    delete[] cpuA;
}