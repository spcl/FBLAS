/**
 * Utility function
 */
#ifndef UTILS_HPP
#define UTILS_HPP
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


/*
 * Data generation functions
 */
void generate_float_vector (float *x, int N)
{
    for(int i=0;i<N;i++)
        x[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0));
}



/*
 * Timing functions
 */


typedef long unsigned int timestamp_t;
inline timestamp_t current_time_usecs() __attribute__((always_inline));
inline timestamp_t current_time_usecs(){
        struct timeval t;
        gettimeofday(&t, NULL);
        return (t.tv_sec)*1000000L + t.tv_usec;

}

/*
 * Test function
 */
inline bool test_equals(float result, float expected, float relative_error)
{
    //check also that the parameters are numbers
    return result==result && expected ==expected && ((result==0 && expected ==0) || fabs(result-expected)/fabs(expected) < relative_error);
}
#endif // UTILS_HPP
