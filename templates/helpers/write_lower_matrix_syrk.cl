/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Writes the matrix C of type {{ helper.type_str }}, results of a SYRK routine.
    Matrix C will be read from channel {{ channels["channel_in_matrix"] }}, row streamed and in tiles
    by rows.

    The value must be accumulated (and multiplied by beta) according to computation requirements.

    The matrix is a lower triangular

    CTILE_COLS reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} * restrict C, const {{ helper.type_str }} beta,const unsigned int N,  const unsigned int ldc)
{

    __constant uint WIDTH = {{ helper.width }};
    __constant uchar CTILE_ROWS = {{ helper.width_y }};
    __constant uchar CTILE_COLS = {{ helper.width_x }};
    __constant ushort MTILE = {{ helper.tile_size }};


    //this kernel will receive the data for C in order
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int InnerBlocksNR = MTILE / CTILE_ROWS;
    const int InnerBlocksNC = MTILE / CTILE_COLS;


    //for each outer tile of C..
    #pragma unroll 1
    for(int ti=0;ti<OuterBlocksN;ti++)
    {
        for(int tj=0;tj<=ti;tj++)
        {
            //load C and multiply it

            //read and save
             #pragma unroll 1
             #pragma ivdep array(C)
             for(int ii=0;ii<InnerBlocksNR;ii++)
                 #pragma unroll 1
                 #pragma ivdep array(C)
                 for(int jj=0;jj<InnerBlocksNC;jj++)
                 {
                     #pragma unroll 1
                     #pragma ivdep array(C)
                     for(int iii=0;iii<CTILE_ROWS;iii++)
                         #pragma unroll
                         for(int jjj=0;jjj<CTILE_COLS;jjj++)
                         {
                                 int ind_i=ti*MTILE+ii*CTILE_ROWS+iii;
                                 int ind_j=tj*MTILE+jj*CTILE_COLS+jjj;
                                 {{ helper.type_str }} c = read_channel_intel({{ channels["channel_in_matrix"] }});
                                 if(ind_i < N && ind_j<=ind_i)
                                     C[ind_i*ldc+ind_j]= beta*C[ind_i*ldc+ind_j] +c;
                         }
                 }
        }
    }
}
