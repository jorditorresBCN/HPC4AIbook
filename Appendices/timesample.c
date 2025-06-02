/* timesample.c */
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define SIZE 1000
typedef double matrix[SIZE][SIZE];
matrix m1;

struct timeval start_time, end_time;

static void foo(void) {
    int i, j;
    for (i = 0; i < SIZE; ++i)
        for (j = 0; j < SIZE; ++j)
            m1[j][i] = 1.0;
}

void print_times() {
    int total_usecs;
    float total_time;
    total_usecs = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                  (end_time.tv_usec - start_time.tv_usec);
    printf(" %.2f mSec \n", ((float) total_usecs) / 1000.0);
    total_time = ((float) total_usecs) / 1000000.0;
}

int main() {
    int i;

    gettimeofday(&start_time, NULL);

    for (i = 0; i < 10; ++i) {
        foo();
    }

    gettimeofday(&end_time, NULL);
    print_times();

    return 0;
}

