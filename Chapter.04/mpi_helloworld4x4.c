#include <stdio.h>
#include <mpi.h>
#include <unistd.h> //  for gethostname

int main (int argc, char **argv) {
 int rank, size;
 MPI_Init(NULL, NULL);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &size);

 char hostname[256];
 gethostname(hostname, sizeof(hostname));
 printf("I am %d of %d running on %s\n", rank, size, hostname);

 MPI_Finalize();
 return 0;
}
