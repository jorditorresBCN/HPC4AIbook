
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 512
#define BLOCK_SIZE_DIM 16

#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

__global__ void matrixProduct(double *matrix_a, double *matrix_b, double *matrix_c, int width) {
    double sum = 0;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col < width && row < width) {
        for (int k = 0; k < width; k++) {
            sum += matrix_a[row * width + k] * matrix_b[k * width + col];
        }
        matrix_c[row * width + col] = sum;
    }
}

void initializeMatrices(double matrix_a[N][N], double matrix_b[N][N]) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j] = rand() % 100;
            matrix_b[i][j] = rand() % 100;
        }
    }
}

void showResults(double matrix_a[N][N], double matrix_b[N][N], double matrix_c[N][N]) {
    printf("***** MATRIX A *****\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            (j % N == N - 1) ? printf("%f\n", matrix_a[i][j]) : printf("%f,", matrix_a[i][j]);
        }
    }
    printf("***** MATRIX B *****\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            (j % N == N - 1) ? printf("%f\n", matrix_b[i][j]) : printf("%f,", matrix_b[i][j]);
        }
    }
    printf("***** MATRIX C *****\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            (j % N == N - 1) ? printf("%f\n", matrix_c[i][j]) : printf("%f,", matrix_c[i][j]);
        }
    }
}

int main() {
    int size = N * N * sizeof(double);
    double h_a[N][N], h_b[N][N], h_c[N][N];
    double *d_a, *d_b, *d_c;

    initializeMatrices(h_a, h_b);

    checkCuda(cudaMalloc((void **) &d_a, size));
    checkCuda(cudaMalloc((void **) &d_b, size));
    checkCuda(cudaMalloc((void **) &d_c, size));

    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    dim3 dimGrid((N + BLOCK_SIZE_DIM - 1) / BLOCK_SIZE_DIM,
                 (N + BLOCK_SIZE_DIM - 1) / BLOCK_SIZE_DIM);
    dim3 dimBlock(BLOCK_SIZE_DIM, BLOCK_SIZE_DIM);
    matrixProduct<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());

    checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    // showResults(h_a, h_b, h_c);

    cudaDeviceReset();
    return 0;
}
