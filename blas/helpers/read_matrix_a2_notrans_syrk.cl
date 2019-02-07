/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type TYPE_T from memory and  push it
    into CHANNEL_MATRIX_A. The matrix is read considering the presence of a
    two level tiling (memory/computational) like the one present in SYRK implementation.

    In this kernel we read the right-most matrix of the computation, non trasposed version
    Matrix A is sent by row
    Each tile-col ti of A is sent a  different number of time, depending on the
    type of SYRK computation:
    - if C is lower triangular, it will be sent ti times
    - if C is upper triangular, it will be sent NumTiles-ti mes


    8 reads are performed simultaneously (this value has been choosen as a trade off between
    generated hardware and speed. In the future can be considered as a parameter).
    If needed, data is padded to tile sizes using zero elements.

*/

__kernel void READ_MATRIX_A2(__global TYPE_T * restrict A, const unsigned int N, const unsigned int K, const unsigned int lda, const unsigned int lower)
{
    //double level of tiling
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int InnerBlocksNR = MTILE / CTILE_ROWS;
    const int InnerBlocksNC = MTILE / CTILE_COLS;
    const int BlocksK=(int)(K/MTILE);
    if(lower!=0 && lower !=1) return;
    int tj_start,tj_end;


    TYPE_T localA2[MTILE];

    for(int ti=0;ti<OuterBlocksN;ti++)
    {
        if(lower==1){
            tj_start=0;
            tj_end=ti;
        }
        else{
            tj_start=ti;
            tj_end=OuterBlocksN-1;
        }
        for(int tj=tj_start;tj<=tj_end;tj++)
        {
            for(int k=0;k<K;k++)
            {
                #pragma unroll 16
                for(int i=0;i<MTILE;i++)
                {
                    if(tj*MTILE+i < N)
                        localA2[i]=A[k*lda+(tj*MTILE+i)];
                    else
                        localA2[i]=0;
                }
                for(int i=0;i<InnerBlocksNR;i++)
                {
                    //iterates over the inner tiles of B
                    for(int ttj=0;ttj<InnerBlocksNC;ttj++)
                    {
                        #pragma unroll
                        for(int j=0;j<CTILE_COLS;j++)
                        {
                            write_channel_intel(CHANNEL_MATRIX_A2,localA2[ttj*CTILE_COLS+j]);
                        }
                    }
                }
            }
        }
    }
}

