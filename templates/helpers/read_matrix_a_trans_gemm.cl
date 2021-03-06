/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a transposed matrix of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_matrix"] }}. The matrix is read considering the presence of a
    two level tiling (memory/computational) like the one present in GEMM implementation.

    In this kernel we read the left-most matrix of the computation (e.g. A in the case of GEMM)
    Each InnerBlock is sent only once (will be buffered by the receiver)
    Matrix A is sent by column (since it is non-transposed)

    CTILE_ROWS reads are performed simultaneously (this value has been choosen as a trade off between
    generated hardware and speed. In the future can be considered as a parameter).
    If needed, data is padded to tile sizes using zero elements.

*/

{% if generate_channel_declaration is defined %}
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel {{ helper.type_str }} {{ channels["channel_out_matrix"] }} __attribute__((depth({{ helper.width }})));
{% endif %}

__kernel void {{ helper_name }}(__global volatile {{ helper.type_str }} * restrict A, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int lda)
{
    //double level of tiling
    
    __constant ushort MTILE = {{ helper.tile_size}};
    __constant uchar CTILE_ROWS = {{ helper.width_y}};
    __constant uchar CTILE_COLS = {{ helper.width_x}};
    
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;
    const int BlocksK=(int)(K/MTILE);

    {{ helper.type_str }} localA[MTILE];
    for(int ti=0; ti< OuterBlocksN;ti++)
    {

        //resend this tile a number of times equal to the number of column tiles of the matrix B
        for(int tj=0;tj<OuterBlocksM;tj++)
        {
            for(int k=0;k<K;k++)
            {
                //load A
                #pragma unroll 16
                for(int i=0;i<MTILE;i++)
                {
                    if(ti*MTILE+i < N)
                        localA[i]=A[k*lda+(ti*MTILE+i)];
                    else
                        localA[i]=0;
                }

                //now we have to iterates over the inner tiles of size CTILE_ROWS x MTILE
                //each of them will be sent only once (and will be reused InnerBlocksM times)
                for(int tti=0; tti<InnerBlocksN;tti++)
                {

                    #pragma unroll
                    for(int i=0;i<CTILE_ROWS;i++)
                    {
                        write_channel_intel({{ channels["channel_out_matrix"] }},localA[tti*CTILE_ROWS+i]);
                    }
                }
            }
        }
    }
}
