#include "laplace2d.hpp"
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>
#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    int n = 1024;
    int max_iters = 1e+6;
    double eps = 1.0e-6;

    boost::program_options::options_description descrip("Allowed options");
    descrip.add_options()
            ("N", boost::program_options::value<int>(&n))
            ("iterations", boost::program_options::value<int>(&max_iters))
            ("error", boost::program_options::value<double>(&eps));

    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), var_map);
    boost::program_options::notify(var_mapm);

    double error = 1.0;

    Laplace A(n+2, n+2);

    A.init();
    std::cout << "Jacobi relaxation Calculation on CPU: " << n << 'x' << n << " mesh\n";

    auto start = std::chrono::high_resolution_clock::now();
    int iter;

    for (iter = 0; error > eps && iter < max_iters; iter++)
    {
        error = A.calcNext();
        A.swap();
        if (iter % 1000 == 0)
            std::cout << iter << ",   " << error << '\n';

        iter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto tm = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << iter << ",   " << error << '\n';
    std::cout << "Elapsed time: " << tm.count() / 1000000. <<std::endl;
    return 0;
}