/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type TYPE_T from memory and push it
    into {{ channels["channel_out_matrix"] }}. This can be used for the systolic implementation of GEMM

    In this kernel we read the left-most B-matrix of the computation
    (i.e. the one that appears in the first matrix-matrix multiplication)
    Each InnerBlock is sent multiple times
    Matrix B is sent by rows (since it is non-transposed)


*/



__kernel void {{ helper_name }}(__global const BASE_TYPE * restrict B, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int ldb)
{

    const uint OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const uint OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const uint InnerBlocksN = MTILE / CTILE_ROWS;
    const uint InnerBlocksM =  MTILE / (CTILE_COLS * VECT_SIZE);
    #pragma loop_coalesce 3
    for(uint ti=0;ti<OuterBlocksN;ti++)
    {
        //outer tile over columns of B
        for(uint tj=0;tj<OuterBlocksM;tj++)
        {
            for(uint k=0;k<K;k++)
            {
                VECT_TYPE localB[MTILE/VECT_SIZE];
                //load it
                for(uint ii=0;ii<(MTILE/VECT_SIZE);ii++){

                    #if VECT_SIZE==1
                        localB[ii] = B[k*ldb+(tj*MTILE+ii)];
                    #elif VECT_SIZE == 2
                        localB[ii] = vload2(0, &B[k*ldb+(tj*MTILE+ii*VECT_SIZE)]);
                    #elif VECT_SIZE == 4
                        localB[ii] = vload4(0, &B[k*ldb+(tj*MTILE+ii*VECT_SIZE)]);
                    #elif VECT_SIZE == 8
                        localB[ii] = vload8(0, &B[k*ldb+(tj*MTILE+ii*VECT_SIZE)]);
                    #elif VECT_SIZE ==16
                        localB[ii] = vload16(0, &B[k*ldb+(tj*MTILE+ii*VECT_SIZE)]);
                    #endif
                    }
                //in case you have to repeat reads
                //then send it
                for(uint i=0;i<InnerBlocksM;i++){
                    #pragma unroll
                    for(uchar j=0;j<CTILE_COLS;j++){
                       write_channel_intel({{ channels["channel_out_matrix"] }}[j],localB[i*CTILE_COLS+j]);
                    }
                }
            }
        }
    }

}