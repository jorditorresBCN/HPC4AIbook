#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    char message[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        // All processes except 0 send a message to process 0
        sprintf(message, "Greetings from process %d!", rank);
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        // Process 0 receives messages from all other processes
        for (int source = 1; source < size; source++) {
            MPI_Recv(message, 100, MPI_CHAR, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 0 received: %s\n", message);
        }
    }

    MPI_Finalize();
    return 0;
}
