/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    GEMM: matrix matrix multiplication with accumulation
    This generic version works for all the four cases (A trans/no transp, B transp/no transp)
    Data arrive always in the same way, it will change the way in which it is read from helpers kernels

    The implementation adopt a two level of tiling: the outermost for the memory (size MTILE x MTILE)
    and the innermost for the computation (CTILE_ROWS x CTILE_COLS).
    The computation is performed by computing outer products. So for each computational tile
    the kernel will receive the corresponding (portion of ) column of A, the row of B
    and will update properly the C elements.

    The kernel computes the matrix C in tiles by rows, Row streamed.
    The results are sent in the channel {{ channels["channel_out_matrix"] }}.
    Matrix A arrive through channel {{ channels["channel_in_matrix_A"] }}, matrix B through channel
    {{ channels["channel_in_matrix_B"] }}. Input  data must be padded to zeros according to the
    reference tile size MTILE.    

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }} __attribute__((depth({{ routine.width_y}})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_B"] }} __attribute__((depth({{ routine.width_x}})));
channel {{ routine.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ routine.tile_size}})));


/**
  Computes the GEMM. A and B are received through input channels.
    - Matrix A: for each outer tile (MTILE size), inner blocks are received one after the other
            The entire outer tile-row (MTILE x K) is resent a number of times equal to the number of
            outer tiles in matrix B (check helpers/read_matrix_a_notrans_gemm.cl for an example)
    - Matrix B: for each outer tile, inner blocks are received one of the other. Each
        outer tile row is sent multiple times (check helpers/read_matrx_b_notrans_gemm.cl for an example)

*/
__kernel void {{ routine.user_name }}(const int N, const int M, const int K, const {{ routine.type_str }} alpha)

{
    __constant ushort MTILE = {{ routine.tile_size}};
    __constant uchar CTILE_ROWS = {{ routine.width_y}};
    __constant uchar CTILE_COLS = {{ routine.width_x}};
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;
    __local {{ routine.type_str }} __attribute__((memory,doublepump)) localC[MTILE/CTILE_ROWS][MTILE/CTILE_COLS][CTILE_ROWS][CTILE_COLS];

    //GEMM with outer product

    for(int ti=0;ti<OuterBlocksN;ti++)	    //outer tile
    {
        for(int tj=0;tj<OuterBlocksM;tj++) //outer tile
        {

            for(int k=0;k<K;k++)
            {
                {{ routine.type_str }} localA[CTILE_ROWS];
                {{ routine.type_str }} localB[CTILE_COLS];
                for(int tti=0;tti<InnerBlocksN;tti++)   //inner tile
                {
                    for(int ttj=0;ttj<InnerBlocksM; ttj++)	//inner tile
                    {
                        if(ttj==0)
                        {
                            #pragma unroll
                            for(int i=0;i<CTILE_ROWS;i++)
                              localA[i]=read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        }
                        #pragma unroll
                        for(int i=0;i<CTILE_COLS;i++)
                            localB[i]=read_channel_intel({{ channels["channel_in_matrix_B"] }});

                        #pragma unroll
                        for(int i=0;i<CTILE_ROWS;i++)
                        {
                            {{ routine.type_str }} tmpa=alpha*localA[i];
                            #pragma unroll
                            for(int j=0; j<CTILE_COLS;j++)
                            {

                                const {{ routine.type_str }} prev=(k==0)?0:localC[tti][ttj][i][j];
                                localC[tti][ttj][i][j]=prev+tmpa*localB[j];
                            }
                        }
                    }
                }
            }

            //prevent unrolls on this, for the sake of saving BRAMs
            #pragma unroll 1
            for(int ii=0;ii<MTILE/CTILE_ROWS;ii++)
            {
                #pragma unroll 1
                for(int jj=0;jj<MTILE/CTILE_COLS;jj++)
                {
                    #pragma unroll 1
                    for(int iii=0;iii<CTILE_ROWS;iii++)
                        #pragma unroll
                        for(int jjj=0;jjj<CTILE_COLS;jjj++)
                        {
                                int ind_i=ti*MTILE+ii*CTILE_ROWS+iii;
                                int ind_j=tj*MTILE+jj*CTILE_COLS+jjj;
                                write_channel_intel({{ channels["channel_out_matrix"] }},localC[ii][jj][iii][jjj]);
                         }
                }
            }
        }
    }
}



