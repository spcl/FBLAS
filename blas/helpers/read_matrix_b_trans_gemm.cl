/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix (transposed) of type TYPE_T from memory and push it
    into CHANNEL_MATRIX_B. The matrix is read considering the presence of a
    two level tiling (memory/computational) like the one present in GEMM implementation.

    In this kernel we read the right-most matrix of the computation (e.g. B in the case of GEMM)
    Each InnerBlock is sent multiple times
    Matrix B is sent by rows

    8 reads are performed simultaneously (tradeoff between generated hardware and performance).
    If needed, data is padded to tile sizes using zero elements.

*/

//read columns of B and send towards column injectors
__kernel void READ_MATRIX_B(__global TYPE_T * restrict B, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int ldb)
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
                #pragma unroll 8
                for(int i=0;i<MTILE;i++)
                {
                    if(tj*MTILE+i < M)
                        localB[i]=B[(tj*MTILE+i)*ldb+k];
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
