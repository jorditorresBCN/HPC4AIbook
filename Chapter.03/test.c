
/* sample.c */
/* Example from PATC course @ BSC*/

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>

#define SIZE 8000
typedef double matrix[SIZE][SIZE];
matrix m1, m2;
struct timeval start_time, end_time;

static void foo (void) {
  int i,j;
  for (i = 0;i < SIZE; ++i)
    for (j = 0; j < SIZE; ++j)
       m1[j][i] = 8.0/m2[j][i];             
}

void check_matrix() {

   int i, j;
   for(i=0;i<SIZE;i++) {
     for(j=0;j<SIZE;j++) {
       if(m1[i][j]!=3.2)
	printf("Check failed\n");
	exit(1);
       }
        //printf("\n");
     }
}

void print_times()
{
   int total_usecs;
   float total_time, total_flops;
   total_usecs = (end_time.tv_sec-start_time.tv_sec) * 1000000 +
         (end_time.tv_usec-start_time.tv_usec);
   printf(" %.2f mSec \n", ((float) total_usecs) / 1000.0);
   total_time = ((float) total_usecs) / 1000000.0;
}

int main () {
  int i,j;

   for(i=0;i<SIZE;i++) {
     for(j=0;j<SIZE;j++) {
	m2[i][j]=2.5;
	}
    }
   gettimeofday(&start_time, NULL);
    for (i = 0; i < 10; ++i) foo();

   gettimeofday(&end_time, NULL);
   print_times();
   check_matrix();
}
