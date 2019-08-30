/*

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type TYPE_T from memory and  push it
    into CHANNEL_MATRIX_A. The matrix is read considering the presence of a
    systolic array for the GEMM implementation.


    In this kernel we read the left-most matrix of the computation (e.g. A in the case of GEMM)
    Each InnerBlock is sent only once (will be buffered by the receiver)
    Matrix A is sent by column (since it is non-transposed)

    8 reads (by defaylt) are performed simultaneously (this value has been choosen as a trade off between
    generated hardware and speed. In the future can be considered as a parameter).
    If needed, data is padded to tile sizes using zero elements.
*/
__kernel void READ_MATRIX_A(__global TYPE_T * restrict A, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int lda)
{
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;
    const int OuterBlocksK = 1 + (int)((K-1) / CHANNEL_UNROLL);
    TYPE_T localA[MTILE][CHANNEL_UNROLL];
    #pragma loop_coalesce 3
    for(int ti=0; ti< OuterBlocksN;ti++) //+1 to flush the last tile of C
    {
        //resend this tile a number of times equal to the number of tile-column of the matrix B
        for(int tj=0;tj<OuterBlocksM;tj++)
        {
            for(int k=0;k<OuterBlocksK;k++)
            {
                //load it
                for(int i=0;i<MTILE;i++)
                {
                    #pragma unroll
                    for(int j=0;j<CHANNEL_UNROLL;j++)
                    {
                        if(ti*MTILE+i < N  && k*CHANNEL_UNROLL+j<K)
                            localA[i][j]=A[(ti*MTILE+i)*lda+k*CHANNEL_UNROLL+j];
                        else
                            localA[i][j]=0;
                    }
                }


                for(int jj=0;jj<CHANNEL_UNROLL;jj++)
                //send it
                for(int i=0;i<MTILE/CTILE_ROWS;i++)
                    if(k*CHANNEL_UNROLL+jj<K){
                        #pragma unroll
                        for(int j=0;j<CTILE_ROWS;j++)
                            write_channel_intel(CHANNEL_MATRIX_A,localA[i*CTILE_ROWS+j][jj]);
                    }
            }
        }
    }

}
