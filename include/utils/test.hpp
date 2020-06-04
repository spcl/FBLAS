#ifndef TEST_HPP
#define TEST_HPP
#include <math.h>


/*
 * utility functions for testing
*/

const double flteps = 1e-4;

inline bool test_equals(float result, float expected, float relative_error)
{
    //check also that the parameters are numbers
    return result==result && expected ==expected && ((result==0 && expected ==0) || fabs(result-expected)/fabs(expected) < relative_error);
}

template <typename T>
bool compare_matrices(T *A, T *B, int n, int m, double flteps){
        bool ok=true;

    //    Measure error by considering nrm_inf
    double nrminf_diff=0, nrminf_orig=0;
    double error;
    bool nan=false;

    for(int i=0;i < n && !nan; i++)
    {
        double nrminf=0, nrminf_o=0;
        for(int j=0; j<m;j++)
        {
            nrminf+=fabs(A[i*m+j]-B[i*m+j]);
            if(isnan(A[i*m+j]) || isnan(B[i*m+j])){
                nan=true;
                printf("NaN identified in position [%d, %d]\n",i,j);
                break;
            }
            nrminf_o+=fabs(B[i*m+j]);
        }
        if(nrminf>nrminf_diff){
            nrminf_diff=nrminf;
        }
        if(nrminf_o>nrminf_orig){
            nrminf_orig=nrminf_o;
        }
    }
    if (nan)
        return false;
    if((nrminf_diff==0 && nrminf_orig ==0) || nrminf_orig==0)
        error=0;
    else
        error=nrminf_diff/nrminf_orig;

    return error<flteps;
}

#endif // TEST_HPP
