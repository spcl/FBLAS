/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Writes the matrix C of type TYPE_T, results of a GEMM routine.
    Matrix C will be read from channel CHANNEL_MATRIX_OUT, row streamed and in tiles
    by rows.

    The value must be accumulated (and multiplied by beta) according to computation requirements.

    CTILE_COLS reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/

__kernel void WRITE_MATRIX(__global volatile TYPE_T * restrict C, const TYPE_T beta,const unsigned int N, const unsigned int M, const unsigned int ldc)
{
    //this kernel will receive the data for C in order
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;

    //__local TYPE_T localC[MTILE][MTILE/CTILE_COLS][CTILE_COLS];


    //for each outer tile of C..
    for(int ti=0;ti<OuterBlocksN;ti++)
    {
        for(int tj=0;tj<OuterBlocksM;tj++)
        {

            //read and save
             #pragma unroll 1
            #pragma ivdep array(C)
             for(int ii=0;ii<MTILE/CTILE_ROWS;ii++)
                 #pragma unroll 1
                 #pragma ivdep array(C)
                 for(int jj=0;jj<MTILE/CTILE_COLS;jj++)
                 {
                     #pragma unroll 1
                     #pragma max_concurrency 1
                     #pragma ivdep array(C)
                     for(int iii=0;iii<CTILE_ROWS;iii++)
                         #pragma unroll
                         for(int jjj=0;jjj<CTILE_COLS;jjj++)
                         {
                                 int ind_i=ti*MTILE+ii*CTILE_ROWS+iii;
                                 int ind_j=tj*MTILE+jj*CTILE_COLS+jjj;
                                 TYPE_T c = read_channel_intel(CHANNEL_MATRIX_OUT);
                                 if(ind_i < N && ind_j<M)
                                     C[ind_i*ldc+ind_j]=beta*C[ind_i*ldc+ind_j]+c;
                         }
                 }
        }
    }


}
