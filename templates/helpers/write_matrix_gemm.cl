/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Writes the matrix C of type {{ helper.type_str }}, results of a GEMM routine.
    Matrix C will be read from channel {{ channels["channel_in_matrix"] }}, row streamed and in tiles
    by rows.

    The value must be accumulated (and multiplied by beta) according to computation requirements.

    CTILE_COLS reads are performed simultaneously.
    Padding data is discarded and not written in memory.

*/


{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_in_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }}  * restrict C, const {{ helper.type_str }} beta,const unsigned int N, const unsigned int M, const unsigned int ldc)
{
    //double level of tiling
    __constant ushort MTILE = {{ helper.tile_size}};
    __constant uchar CTILE_ROWS = {{ helper.width_y}};
    __constant uchar CTILE_COLS = {{ helper.width_x}};

    //this kernel will receive the data for C in order
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;

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
                                 {{ helper.type_str }} c = read_channel_intel({{ channels["channel_in_matrix"] }});
                                 if(ind_i < N && ind_j<M)
                                     C[ind_i*ldc+ind_j]=beta*C[ind_i*ldc+ind_j]+c;
                         }
                 }
        }
    }


}
