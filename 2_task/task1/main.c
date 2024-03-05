#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <malloc.h>
#include <time.h>

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n);

int num_of_threads = 0;

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n){
    #pragma omp parallel num_threads(num_of_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }  
    }
}

void run_parallel(int m, int n){
    double *a, *b, *c;
    // Allocate memory for 2-d array a[m, n]
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);
    #pragma omp parallel num_threads(num_of_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                a[i * n + j] = i + j;
            c[i] = 0.0;
        }
    }
    for (int j = 0; j < n; j++)
        b[j] = j;
    
    //GET TIME
    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime() - t;
    //GET TIME

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv)
{
    //printf("%s\n", argv[1]);
    num_of_threads = atoi(argv[1]);
    //printf("%d\n", num_of_threads);
    int m = 40000, n = 40000;
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    run_parallel(m, n);
    return 0;
}