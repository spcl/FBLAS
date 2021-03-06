/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
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

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


__kernel void {{ routine.user_name }}(const int N, const int M, const {{ routine.type_str }} alpha, __global const {{ routine.type_str }} * restrict A,
                          const unsigned int lda, __global {{ routine.type_str }} * restrict B, const unsigned int ldb )
{

    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_M = {{ routine.tile_m_size }};

    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int computing_outer_loop_limit=(int)(TILE_M/WIDTH);


    for (int i = N-1; i >= 0 ; i--)
    {

        const {{ routine.type_str }} aii=A[i*lda+i];
        for(int tj=0;tj<BlocksM;tj++)
        {
            {{ routine.type_str }} localB[TILE_M];

            for(int jj=0;jj<computing_outer_loop_limit;jj++)
            {
                #pragma unroll
                for(int j=0;j<WIDTH;j++)
                {
                    if(tj*TILE_M+jj*WIDTH+j < M)
                    {
                        const {{ routine.type_str }} prev=(i==N-1)?alpha*B[i*ldb+tj*TILE_M+jj*WIDTH+j]:B[i*ldb+tj*TILE_M+jj*WIDTH+j];
                        localB[jj*WIDTH+j]=prev/aii;
                        B[i*ldb+tj*TILE_M+jj*WIDTH+j]=localB[jj*WIDTH+j]; //save the result
                    }
                }
            }
            #pragma ivdep array(B) //false dependency on B (each row of B will be touched only once)
            for(int k=0;k<i;k++)
            {
                {{ routine.type_str }} aik=A[i*lda+k];
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {
                    const int off_idx=k*ldb+tj*TILE_M+jj*WIDTH;

                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                    {
                        if(tj*TILE_M+jj*WIDTH+j<M)
                        {
                            const {{ routine.type_str }} prev=(i==N-1)?alpha*B[k*ldb+tj*TILE_M+jj*WIDTH+j]:B[k*ldb+tj*TILE_M+jj*WIDTH+j];
                            B[k*ldb+tj*TILE_M+jj*WIDTH+j] = prev - aik * localB[jj*WIDTH+j];
                        }
                    }
                }
            }

        }


    }
}
