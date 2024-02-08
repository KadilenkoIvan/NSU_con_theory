#include <iostream>
#include <cmath>

#define SIZE 10000000

#ifdef FLOAT
    #define TYPE float
#else
    #define TYPE double
#endif

using curr_type = TYPE;

int main()
{
    curr_type pi = 3.14159265358979323846;
    curr_type* arr = new curr_type[SIZE];
    curr_type sum = 0;
    curr_type angle = 0;
    for (int i = 0; i < SIZE; ++i) {
        angle = (i * 2 * pi) / SIZE;
        arr[i] = std::sin(angle);
        sum += arr[i];
    }
    std::cout << sum;
}