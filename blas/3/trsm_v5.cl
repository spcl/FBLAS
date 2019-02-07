/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    TRSM v5
    TRSM  solves one of the matrix equations

        op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

     where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

        op( A ) = A   or   op( A ) = A**T.

     The matrix X is overwritten on B.

     This version solves the case: side right, A lower, non transposed.
     No distinction is made between unit/non unit cases

*/

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dtrsm
#define TILE_M 32
#define W 16
#define KERNEL_NAME trsm_v5
#define __STRATIX_10__

//FBLAS_PARAMETERS_END
#include <commons.h>

__kernel void KERNEL_NAME(const int N, const int M, const TYPE_T alpha, __global const TYPE_T * restrict A,
                          const unsigned int lda, __global TYPE_T * restrict B, const unsigned int ldb )
{

    const int BlocksM=1+(int)((M-1)/TILE_M);
    //TODO this implementation deserves tiling, but this will change drammatically the logic

    for(int i=0;i<N;i++)
    {
        for(int j=M-1; j>= 0; j--)
        {
            const TYPE_T Ajj=A[j*lda+j];
            const TYPE_T prev=(j==M-1)?alpha*B[i*ldb+j]:B[i*ldb+j];
            const TYPE_T Bij = prev/Ajj;
            B[i*ldb+j]=Bij;

            const int computing_outer_loop_limit=1+(int)((j-1)/W);

            for(int tk=0;tk < computing_outer_loop_limit; tk++)
            {
                #pragma unroll
                for(int k=0; k < W ;k++)
                {
                    if(tk*W+k<j)
                    {
                        const TYPE_T prev=(j==M-1)?alpha*B[i*ldb+tk*W+k]:B[i*ldb+tk*W+k];
                        B[i*ldb+tk*W+k]=prev-A[j*lda+tk*W+k]*Bij;
                    }
                }
            }

        }
    }
}
