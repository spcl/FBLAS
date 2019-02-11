/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type TYPE_T from memory and push it
    into CHANNEL_MATRIX_B. This can be used for implementin SYR2K routine

    In this kernel we read the left-most B-matrix of the computation
    (i.e. the one that appears in the first matrix-matrix multiplication)
    Each InnerBlock is sent multiple times
    Matrix B is sent by rows (since it is non-transposed)

    CTILE_COLS reads are performed simultaneously.
    If needed, data is padded to tile sizes using zero elements.

*/

__kernel void READ_MATRIX_B(__global volatile TYPE_T * restrict B, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int ldb)
{

    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;

    TYPE_T localB[MTILE];

    for(int ti=0;ti<OuterBlocksN;ti++)
    {
        //outer tile over columns of B
        for(int tj=0;tj<OuterBlocksM;tj++)
        {
            for(int k=0;k<K;k++)
            {
                #pragma unroll 16
                for(int i=0;i<MTILE;i++)
                {
                    if(tj*MTILE+i < M)
                        localB[i]=B[k*ldb+(tj*MTILE+i)];
                    else
                        localB[i]=0;
                }
                for(int i=0;i<InnerBlocksN;i++)
                {
                    //iterates over the inner tiles of B
                    for(int ttj=0;ttj<InnerBlocksM;ttj++)
                    {
                        #pragma unroll
                        for(int j=0;j<CTILE_COLS;j++)
                        {
                            write_channel_intel(CHANNEL_MATRIX_B,localB[ttj*CTILE_COLS+j]);
                        }
                    }
                }
            }
        }
    }
}
