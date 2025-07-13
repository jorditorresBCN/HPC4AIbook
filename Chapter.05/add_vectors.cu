#include <stdio.h>

#define N 64 
#define THREADS_PER_BLOCK 32 

void init_vector(int n, int *V) {
    for (int i = 0; i < n; i++) {
         V[i]=i;
    }
}

void print_vector(int n, int *V) {
    for (int i = 0; i < n; i++) {
        printf("|%d", V[i]);
    }
    printf("\n");
}


__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}


int main(void) {
    int *a, *b, *c;             // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c 
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    init_vector(N,a);
    init_vector(N,b);

    printf("vector a:\n");
    print_vector(N, a);
    printf("vector b:\n");
    print_vector(N, b);


    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


    add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a,d_b,d_c,N);
//    add<<< ... >>>(...);


    // Copy result back to the host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("vector c:\n");
    print_vector(N, c);

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
