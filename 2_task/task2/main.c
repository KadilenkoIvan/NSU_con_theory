#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <malloc.h>
#include <time.h>
#include <math.h>

int num_of_threads = 0;

double integrate_omp(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel num_threads(num_of_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0;
        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));
    
        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

const double PI = 3.14159265358979323846;
const double a = -20.0;
const double b = 20.0;
const long long nsteps = 40000000;

double run_parallel(double (*func)(double)){
    double t = omp_get_wtime();
    double res = integrate_omp(func, a, b, nsteps);
    t = omp_get_wtime() - t;
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double func(double x){
    return pow(x, 2);
}

int main(int argc, char **argv){
    num_of_threads = atoi(argv[1]);
    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tparallel = run_parallel(func);
    printf("Execution time (parallel): %.6f\n", tparallel);
    return 0;
}