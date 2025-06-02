#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

double local_trap(double a, double b, long long int local_n) {
    double h = (b - a) / local_n;
    double sum = (f(a) + f(b)) / 2.0;
    for (long long int i = 1; i < local_n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

int main(int argc, char** argv) {
    int rank, size;
    long long int n = 4294967296LL;  // 16^8
    double a = 0.0, b = 1.0;
    double local_a, local_b, h;
    long long int local_n;
    double local_result, total_result;
    struct timeval start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = (n / size) * size;
    h = (b - a) / n;
    local_n = n / size;
    local_a = a + rank * local_n * h;
    local_b = local_a + local_n * h;

    if (rank == 0) gettimeofday(&start, NULL);

    local_result = local_trap(local_a, local_b, local_n);

    if (rank == 0) {
        total_result = local_result;
        for (int source = 1; source < size; source++) {
            double temp;
            MPI_Recv(&temp, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_result += temp;
        }
        gettimeofday(&end, NULL);
        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                         (end.tv_usec - start.tv_usec) / 1000.0;
        printf("Estimated PI = %.16f\n", total_result);
        printf("Parallel execution time with %d processes: %.2f ms\n", size, elapsed);
    } else {
        MPI_Send(&local_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
