/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type {{ helper.type_str }} from memory and  push it
    into {{ channels["channel_out_matrix"] }}.  The matrix is read considering the presence of a
    systolic array for the GEMM implementation.


    In this kernel we read the left-most matrix of the computation (e.g. A in the case of GEMM)
    Each InnerBlock is sent only once (will be buffered by the receiver)
    Matrix A is sent by column (since it is non-transposed)

    8 reads (by defaylt) are performed simultaneously (this value has been choosen as a trade off between
    generated hardware and speed. In the future can be considered as a parameter).
    If needed, data is padded to tile sizes using zero elements.
*/

__kernel void {{ helper_name }}(__global volatile const BASE_TYPE * restrict A, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int lda)
{
    const uint OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const uint OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const uint InnerBlocksN = MTILE / CTILE_ROWS;
    const uint InnerBlocksM =  MTILE / (CTILE_COLS*VECT_SIZE);
    const uint BlocksK=1 + (int)((K-1) / CHANNEL_UNROLL);
    BASE_TYPE localA[MTILE][CHANNEL_UNROLL];
    const uint flattened_loop_limit = OuterBlocksN * OuterBlocksM * BlocksK;
    const uint iterations_for_draining =  (MTILE * MTILE / (CTILE_COLS*VECT_SIZE))/(CHANNEL_UNROLL*InnerBlocksN*InnerBlocksM);
    uint ti=0, tj =0, k=0;
    int num_send_A=0;
    //flattened loop
    for(int it =0; it< flattened_loop_limit + iterations_for_draining; it++){
        //load it
        for(uint ii=0;ii<MTILE;ii++)
        {
            BASE_TYPE value;
            #pragma unroll
            for(uchar j=0;j<CHANNEL_UNROLL;j++)
            {
                if(ti*MTILE+ii < N  && k*CHANNEL_UNROLL+j<K)
                {
                    value=A[(ti*MTILE+ii)*lda+k*CHANNEL_UNROLL+j];
                }
                else
                    value=0;
                //value = __fpga_reg(value);
                //localA[ii][j] = __fpga_reg(localA[ii][j]);
                localA[ii][j] = value;
            }

        }

        #pragma loop_coalesce
        for(uchar jj=0;jj<CHANNEL_UNROLL;jj++)
            //send it
            for(uint i=0;i<InnerBlocksN;i++){
                 if(k * CHANNEL_UNROLL +jj <K){
                    // float value = 0;
                    #pragma unroll
                    for(uchar j=0;j<CTILE_ROWS;j++){
                        BASE_TYPE value =localA[i*CTILE_ROWS+j][jj];
                        //value =__fpga_reg(value);
                        write_channel_intel({{ channels["channel_out_matrix"] }}[j],value);
                    }
                 }
            }
        //k goes from 0 to blocksK
        k = ((k != BlocksK-1) ? k+1 : 0);
        //tj goes from 0 to outerBlocksM (only when k advances)
        tj = ((k == 0) ? ((tj != OuterBlocksM -1)? tj+1 : 0) : tj);

        //ti goes from 0 to OuterBlocksN+1 (we have additional iterations)
        ti = ((k == 0 && tj == 0)? ti+1 : ti);
    }
}