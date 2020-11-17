/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    SYRK: symmetric rank update
    This generic version works for all the four cases (A trans/no tranc, C lower/ upper)

    This kernel receives the data always in the same way. The data will be read by helper
    kernels in different way, according to the desired computation.
    The data related to matrix A (or A tranposed) arrives in channel {{ channels["channel_in_matrix_A"] }}.
    The data of the matrix A tranposed (or A) arrives in channel {{ channels["channel_in_matrix_A2"] }}.
    Results are produce in {{ channels["channel_out_matrix"] }}, in tiles by row, row streamed.
    Only the meaninful tiles are produced (i.e. the lower or the upper ones)

    Computationally, this kernel resembles the GEMM implementation.
    A 2-level tiling is applied.



*/




#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}

channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }} __attribute__((depth({{ routine.width_y}})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A2"] }} __attribute__((depth({{ routine.width_x}})));
channel {{ routine.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ routine.tile_size}})));


/*
    This kernel computes  C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
    where C can be triangular lower or upper. A (the first A) can be transposed (size KxN)
    or not (size NxK).
    The elements of the two orientation of A are received by two channels {{ channels["channel_in_matrix_A"] }}
    and {{ channels["channel_in_matrix_A2"] }}, with which we refer to the first or second A accordingly

    The output is sent to a writer, produced row streamed and tiles by row (only the meaningful tiles)
    A 2 level tiling is applied

    To drive the computation, the kernel receives also an additional parameters:
    - lower, 1 if C is a lower matrix, 0 otherwise


*/
__kernel void {{ routine.user_name }}(const {{ routine.type_str }} alpha,  const unsigned int N, const unsigned int K, const unsigned int lower)
{

    __constant ushort MTILE = {{ routine.tile_size }};
    __constant uchar CTILE_ROWS = {{ routine.width_y }};
    __constant uchar CTILE_COLS = {{ routine.width_x }};
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int InnerBlocksNR = MTILE / CTILE_ROWS;
    const int InnerBlocksNC = MTILE / CTILE_COLS;

    __local {{ routine.type_str }} __attribute__((memory,doublepump)) localC[MTILE/CTILE_ROWS][MTILE/CTILE_COLS][CTILE_ROWS][CTILE_COLS];


    //Adjust the loop iteration space of the second loop depending if C is lower or upper matrix
    int tj_start,tj_end;

    for(int ti=0;ti<OuterBlocksN;ti++)	    //outer tile
    {
        if(lower==1)
        {
            tj_start=0;
            tj_end=ti;
        }
        else
        {
            tj_start=ti;
            tj_end=OuterBlocksN-1;
        }
        for(int tj=tj_start;tj<=tj_end;tj++) //C is lower triangular
        {
            for(int k=0;k<K;k++)
            {
                    //buffer A
                    {{ routine.type_str }} localAi[CTILE_ROWS];
                    {{ routine.type_str }} localAj[CTILE_COLS];

                    //compute
                    #pragma unroll 1
                    for(int tti=0;tti<InnerBlocksNR;tti++)   //inner tile
                    {
                            //const int inner_block_limit=(tj<ti)?InnerBlocksN : tti;
                        #pragma unroll
                        for(int i=0;i<CTILE_ROWS;i++)
                            localAi[i]=read_channel_intel({{ channels["channel_in_matrix_A"] }});
                        #pragma unroll 1
                        for(int ttj=0;ttj<InnerBlocksNC; ttj++)	//inner tile
                        {

                            #pragma unroll
                            for(int i=0;i<CTILE_COLS;i++)
                                localAj[i]=read_channel_intel({{ channels["channel_in_matrix_A2"] }});

                            //to unroll
                            #pragma unroll
                            for(int i=0;i<CTILE_ROWS;i++)
                            {

                                {{ routine.type_str }} tmpa=alpha*localAi[i];
                                #pragma unroll
                                for(int j=0; j<CTILE_COLS;j++)
                                {

                                    const {{ routine.type_str }} prev=(k==0)? 0:localC[tti][ttj][i][j];
                                    localC[tti][ttj][i][j]=prev+tmpa*localAj[j];
                                }

                            }
                        }
                    }

            }

            //prevent unrolls on this, for the sake of saving BRAMs
            #pragma unroll 1
            for(int ii=0;ii<InnerBlocksNR;ii++)
                #pragma unroll 1
                for(int jj=0;jj<InnerBlocksNC;jj++)
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

