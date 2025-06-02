#include <stdio.h>

double f(double x) {
    return 4.0 / (1.0 + x * x);
}

double trapezoidal_rule(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = (f(a) + f(b)) / 2.0;

    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }

    return sum * h;
}

int main() {
    int n = 1000000;
    double a = 0.0, b = 1.0;
    double pi = trapezoidal_rule(a, b, n);
    printf("Estimated PI = %.16f\n", pi);
    return 0;
}
