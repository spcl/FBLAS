
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    TRSM v3
    TRSM  solves one of the matrix equations

        op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

     where alpha is a scalar, X and B are m by n matrices, A is a unit, or
     non-unit,  upper or lower triangular matrix  and  op( A )  is one  of

        op( A ) = A   or   op( A ) = A**T.

     The matrix X is overwritten on B.

     This version solves the case: side left, A lower, transposed.
     No distinction is made between unit/non unit cases

*/

//FBLAS_PARAMETERS_START

//#define DOUBLE_PRECISION		//enable if dtrsm
#define TILE_M 32
#define W 8
#define KERNEL_NAME trsm_v3
#define __STRATIX_10__
//FBLAS_PARAMETERS_END

#include <commons.h>


__kernel void KERNEL_NAME(const int N, const int M, const TYPE_T alpha, __global const TYPE_T * restrict A,
                          const unsigned int lda, __global TYPE_T * restrict B, const unsigned int ldb )
{
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int computing_outer_loop_limit=(int)(TILE_M/W);


    for (int i = N-1; i >= 0 ; i--)
    {

        const TYPE_T aii=A[i*lda+i];
        for(int tj=0;tj<BlocksM;tj++)
        {
            TYPE_T localB[TILE_M];

            for(int jj=0;jj<computing_outer_loop_limit;jj++)
            {
                #pragma unroll
                for(int j=0;j<W;j++)
                {
                    if(tj*TILE_M+jj*W+j < M)
                    {
                        const TYPE_T prev=(i==N-1)?alpha*B[i*ldb+tj*TILE_M+jj*W+j]:B[i*ldb+tj*TILE_M+jj*W+j];
                        localB[jj*W+j]=prev/aii;
                        B[i*ldb+tj*TILE_M+jj*W+j]=localB[jj*W+j]; //save the result
                    }
                }
            }
            #pragma ivdep array(B) //false dependency on B (each row of B will be touched only once)
            for(int k=0;k<i;k++)
            {
                TYPE_T aik=A[i*lda+k];
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {
                    const int off_idx=k*ldb+tj*TILE_M+jj*W;

                    #pragma unroll
                    for(int j=0;j<W;j++)
                    {
                        if(tj*TILE_M+jj*W+j<M)
                        {
                            const TYPE_T prev=(i==N-1)?alpha*B[k*ldb+tj*TILE_M+jj*W+j]:B[k*ldb+tj*TILE_M+jj*W+j];
                            B[k*ldb+tj*TILE_M+jj*W+j] = prev - aik * localB[jj*W+j];
                        }
                    }
                }
            }

        }


    }
}
