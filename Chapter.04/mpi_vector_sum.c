
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 16  // tamaño total del vector

int main(int argc, char** argv) {
    int rank, size;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = N / size;  // asumimos que N divisible por número de procesos

    int* full_vector = NULL;
    int* local_vector = malloc(local_n * sizeof(int));
    int* result_vector = NULL;

    if (rank == 0) {
        full_vector = malloc(N * sizeof(int));
        result_vector = malloc(N * sizeof(int));
        for (i = 0; i < N; i++) {
            full_vector[i] = i + 1;  // ejemplo: [1, 2, 3, ..., N]
        }
        printf("Root process: initial vector = ");
        for (i = 0; i < N; i++) printf("%d ", full_vector[i]);
        printf("\n");
    }

    // Distribuir el vector entre los procesos
    MPI_Scatter(full_vector, local_n, MPI_INT,
                local_vector, local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    printf("Process %d received elements:", rank);
    for (i = 0; i < local_n; i++) printf(" %d", local_vector[i]);
    printf("\n");

    // Operación local: sumar 1 a cada elemento
    for (i = 0; i < local_n; i++) {
        local_vector[i] += 1;
    }

    // Recolectar los resultados en el root
    MPI_Gather(local_vector, local_n, MPI_INT,
               result_vector, local_n, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Root process: gathered result = ");
        for (i = 0; i < N; i++) printf("%d ", result_vector[i]);
        printf("\n");
    }

    // Limpieza
    if (rank == 0) {
        free(full_vector);
        free(result_vector);
    }
    free(local_vector);

    MPI_Finalize();
    return 0;
}
