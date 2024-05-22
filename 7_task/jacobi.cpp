#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include "laplace2d.hpp"
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char **argv)
{
    int n = 1024;
    int max_iters = 1e+6;
    double eps = 1.0e-6;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("N", po::value<int>(&n))
            ("iterations", po::value<int>(&max_iters))
            ("error", po::value<double>(&eps));
    po::variables_map var_map;
    po::store(po::parse_command_line(argc, argv, desc), var_map);
    po::notify(var_map);
	n+=2;

    double error = 1.0;

    Laplace A(n, n);

    A.init();
    std::cout << "Jacobi relaxation Calculation cuda: " << n << 'x' << n << " mesh\n";

    auto start = std::chrono::high_resolution_clock::now();
    int iter;
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (iter = 0; error > eps && iter < max_iters; iter++)
    {
        error = A.calcNext(handle);
        A.swap();
        if (iter % 100 == 0)
            std::cout << iter << ",   " << error << '\n';
        iter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << iter << ",   " << error << '\n';
    std::cout << "Elapsed time: " << tm.count() / 1000000. <<std::endl;
    return 0;
}