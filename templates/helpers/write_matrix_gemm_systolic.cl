/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Writes the matrix C of type {{ helper.type_str }}, results of a GEMM routine.
    Matrix C will be read from channel {{ channels["channel_in_matrix"] }}, row streamed and in tiles
    by rows.

    Data arrives from a systolic GEMM.

*/

__kernel void {{ helper_name }}(__global volatile BASE_TYPE * restrict C, const BASE_TYPE beta,const unsigned int N, const unsigned int M, const unsigned int ldc)
{
    //this kernel will receive the data for C in order
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / (CTILE_COLS * VECT_SIZE);
    int received=0;

    //for each outer tile of C, receive and accumulate
    #pragma loop_coalesce
    #pragma ivdep array(C)
    //#pragma speculated_iterations 0
    for(uint ti=0;ti<OuterBlocksN;ti++)
    {

        #pragma ivdep array(C)
        for(uint tj=0;tj<OuterBlocksM;tj++)
        {
            //read and save
            #pragma ivdep array(C)
            for(uchar iii=0;iii<CTILE_ROWS;iii++)
            {
                #pragma ivdep array(C)
                for(uint ii=0;ii<InnerBlocksN;ii++)
                {

                    #pragma ivdep array(C)
                    for(uint jj=0;jj<InnerBlocksM;jj++)
                    {
                        ctile_col_drain res;//= uno;
                        #pragma ivdep array(C)
                        for(uchar jjj=0;jjj<CTILE_COLS/WRITES_UNROLLED;jjj++)
                        {
                            if(jjj==0) res=  read_channel_intel({{ channels["channel_in_matrix"] }});
                            #pragma unroll
                            for(uchar wr=0;wr < WRITES_UNROLLED;wr++)
                            {
                                #pragma unroll
                                for(uchar vj = 0; vj < VECT_SIZE; vj++){
                                     int ind_i=ti*MTILE+ii*CTILE_ROWS+iii;
                                     int ind_j=tj*MTILE+jj*CTILE_COLS*VECT_SIZE+jjj*WRITES_UNROLLED*VECT_SIZE+wr*VECT_SIZE+vj;
                                      #if VECT_SIZE == 1
                                         if(ind_i < N && ind_j<M)
                                          C[ind_i*ldc+ind_j]=beta*C[ind_i*ldc+ind_j]+res.drain_data[jjj*WRITES_UNROLLED+wr];
                                      #else
                                      if(ind_i < N && ind_j<M){
                                          C[ind_i*ldc+ind_j]=beta*C[ind_i*ldc+ind_j]+res.drain_data[jjj*WRITES_UNROLLED+wr][vj];
                                        }

                                      #endif
                                }
                            }
                        }
                    }
                 }
            }
        }
    }
}
