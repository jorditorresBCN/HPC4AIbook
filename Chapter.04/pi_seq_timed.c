#include <stdio.h>
#include <sys/time.h>

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

double trapezoidal_rule(double a, double b, long long int n) {
    double h = (b - a) / n;
    double sum = (f(a) + f(b)) / 2.0;
    for (long long int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

int main() {
    struct timeval start, end;
    long long int n = 4294967296LL;  // 16^8
    double a = 0.0, b = 1.0;

    gettimeofday(&start, NULL);
    double pi = trapezoidal_rule(a, b, n);
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Estimated PI = %.16f\n", pi);
    printf("Sequential execution time: %.2f ms\n", elapsed);
    return 0;
}
