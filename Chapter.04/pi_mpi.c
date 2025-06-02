#include <stdio.h>
#include <mpi.h>

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

double local_trap(double a, double b, int local_n) {
    double h = (b - a) / local_n;
    double sum = (f(a) + f(b)) / 2.0;

    for (int i = 1; i < local_n; i++) {
        sum += f(a + i * h);
    }

    return sum * h;
}

int main(int argc, char** argv) {
    int rank, size, n;
    double a = 0.0, b = 1.0;
    double local_a, local_b, h;
    int local_n;
    double local_result, total_result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensuring that n is divisible by the number of MPI processes
    n = 1000000
    n = (n / size) * size; 


    h = (b - a) / n;
    local_n = n / size;
    local_a = a + rank * local_n * h;
    local_b = local_a + local_n * h;

    local_result = local_trap(local_a, local_b, local_n);

    if (rank == 0) {
        total_result = local_result;
        for (int source = 1; source < size; source++) {
            double temp;
            MPI_Recv(&temp, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_result += temp;
        }
        printf("Estimated PI = %.16f\n", total_result);
    } else {
        MPI_Send(&local_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
} 

