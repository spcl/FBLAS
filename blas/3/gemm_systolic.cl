/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    GEMM: matrix matrix multiplication, systolic implementation with accumulation
    This generic version works for all the four cases (A trans/no transp, B transp/no transp)
    Data arrive always in the same way, it will change the way in which it is read from helpers kernerls

    The systolic array is implemented taking inspiration from Intel documentation

    The kernel computes the matrix C in tiles by rows, Row streamed.
    The results are sent in the channel CHANNEL_MATRIX_OUT.
    Matrix A arrive through channel CHANNEL_MATRIX_A, matrix B through channel
    CHANNEL_MATRIX_B. Input  data must be padded to zeros according to the
    reference tile size MTILE.

    Result is streamed in an output channel, tile by tile as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable
#define CHANNEL_UNROLL 8

//FBLAS_PARAMETERS_START
//#define DOUBLE_PRECISION		//enable if dgemm

#define CTILE_ROWS 8
#define CTILE_COLS 8
#define MTILE 128
//namigs
#define CHANNEL_MATRIX_A channel_A
#define CHANNEL_MATRIX_B channel_B
#define CHANNEL_MATRIX_OUT channel_C
#define KERNEL_NAME gemm

//architecture
#define __STRATIX_10__

//FBLAS_PARAMETERS_END

#include <commons.h>

#if defined(DOUBLE_PRECISION)
typedef struct  {
    double drain_data[CTILE_COLS];
} ctile_col_double;
#define RES_TYPE_T ctile_col_double
#else
typedef struct  {
    float drain_data[CTILE_COLS];
} ctile_col_float;
#define RES_TYPE_T ctile_col_float
#endif
channel float CHANNEL_MATRIX_A __attribute__((depth(MTILE)));
channel float CHANNEL_MATRIX_B __attribute__((depth(MTILE)));
channel  RES_TYPE_T CHANNEL_MATRIX_OUT __attribute__((depth(MTILE)));
__constant int SHIFT_REG_SIZE=MTILE/CTILE_ROWS*MTILE/CTILE_COLS;

//compute function
float PE(int k, int tti, int ttj, int i, int j, float a_reg[CTILE_ROWS][CTILE_COLS+1] ,     
            float b_reg[CTILE_ROWS+1][CTILE_COLS], float *accum)
{
    a_reg[i][j+1]=__fpga_reg(__fpga_reg(a_reg[i][j]));
    b_reg[i+1][j]=__fpga_reg(__fpga_reg(b_reg[i][j]));
    float oldAcc= __fpga_reg(accum[0]);
    float prev=(k==0)?0:oldAcc;
    prev+=a_reg[i][j]*b_reg[i][j];;
    #pragma unroll
    for (int i = 0; i < SHIFT_REG_SIZE - 1; i++) {
        accum[i] = accum[i + 1];
    }
    accum[SHIFT_REG_SIZE-1]=prev;
    return prev;
}

__kernel void KERNEL_NAME(const int N, const int M, const int K, const float alpha)

{
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int numCBlocks=OuterBlocksN*OuterBlocksM;
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;

    //shift registers for draining
    __private float  draining[CTILE_COLS][(CTILE_ROWS-1)*SHIFT_REG_SIZE+1];
    float accum[CTILE_ROWS][CTILE_COLS][SHIFT_REG_SIZE];
    float  localB[MTILE];

    int elements_to_send=0;     //how many elements we have to send for each iteration of the while loop (MTILE*MTILE)
    int computedCBlocks=0;      //counter used for iteration of the while loop in which we don't compute but just drain
    while (1)
    {
        //on each of this iteration, we compute one block of C (MTILE*MTILE)
        #pragma loop_coalesce
        #pragma ivdep               //ignore ghost dependencies
        for(int k=0;k<K;k++)
        {
            for(int tti=0;tti<InnerBlocksN;tti++)
            {
                float a_reg[CTILE_ROWS][CTILE_COLS+1];
                float b_reg[CTILE_ROWS+1][CTILE_COLS];

                for(int ttj=0;ttj<InnerBlocksM; ttj++)
                {

                    //receive A and B: channels read are executed only if there are still blocks of C
                    //that must be computed. Otherwise we are draining the last block of C
                    float fedB[CTILE_COLS];

                    if(ttj==0 && computedCBlocks<numCBlocks)    //Load A. CTILE_ROWS element of A are reused
                        #pragma unroll                          //throughout the loop over ttj
                        for(int i=0;i<CTILE_ROWS;i++){
                            float tmp=alpha*read_channel_intel(CHANNEL_MATRIX_A);
                            a_reg[i][0]=tmp;
                            a_reg[i][0]=__fpga_reg(__fpga_reg(a_reg[i][0]));
                        }

                    if(tti==0 && computedCBlocks<numCBlocks)
                    {
                        #pragma unroll                          //Load B, sving it for successive
                        for(int j=0;j<CTILE_COLS;j++)           //reuse
                        {
                            localB[ttj*CTILE_COLS+j]=read_channel_intel(CHANNEL_MATRIX_B);
                            localB[ttj*CTILE_COLS+j]=__fpga_reg(__fpga_reg(localB[ttj*CTILE_COLS+j]));
                        }
                    }

                    #pragma unroll                          //reuse B
                    for(int j=0;j<CTILE_COLS;j++)
                    {
                        b_reg[0][j]=localB[ttj*CTILE_COLS+j];
                        b_reg[0][j]=__fpga_reg(__fpga_reg(b_reg[0][j]));
                    }



                    //send drained elements to write kernel
                    if(k==K-1 && tti==0 && ttj==0 )
                        elements_to_send=MTILE*MTILE;           //reset when we finished to compute a tile of C

                    #pragma unroll
                    for(int i=0;i<CTILE_ROWS;i++)
                    {

                        #pragma unroll
                        for(int j=0; j<CTILE_COLS;j++)
                        {
                            float res=PE(k,tti,ttj,i,j,a_reg,b_reg, accum[i][j]);
                            if(k==K-1) //data needs to be evacuated. Each PE writes always in the  same position
                                draining[j][i*SHIFT_REG_SIZE]=res;
                            fedB[j]=__fpga_reg(__fpga_reg(fedB[j])); //apparently, this help synthesis

                        }
                    }

                    //build the drained result
                    RES_TYPE_T drained;

                    #pragma unroll
                    for(int j=0;j<CTILE_COLS;j++)
                    {
                        drained.drain_data[j]=draining[j][0];
                        #pragma unroll
                        for(int jj=0;jj<CTILE_COLS;jj++)
                            drained.drain_data[jj]=__fpga_reg(__fpga_reg(drained.drain_data[jj]));
                    }


                    //send to writer (use coalesced writes)
                    if(elements_to_send>0 && computedCBlocks <=numCBlocks)
                    {
                        write_channel_intel(CHANNEL_MATRIX_OUT,drained);
                        elements_to_send-=CTILE_COLS;
                    }


                    //shift all the draining registers
                    #pragma unroll
                    for(int jj=0;jj<CTILE_COLS;jj++){

                        #pragma unroll
                        for(int i=0;i<CTILE_ROWS-1;i++)
                        {
                            #pragma unroll
                            for(int ii=0;ii<SHIFT_REG_SIZE-1;ii++)
                                 draining[jj][i*SHIFT_REG_SIZE+ii]=draining[jj][i*SHIFT_REG_SIZE+ii+1];

                            draining[jj][i * SHIFT_REG_SIZE + SHIFT_REG_SIZE - 1] = __fpga_reg(__fpga_reg(draining[jj][i * SHIFT_REG_SIZE + SHIFT_REG_SIZE]));

                        }
                    }
                }
            }
        }
        computedCBlocks++;
        if(computedCBlocks>numCBlocks) //we finished this GEMM
            computedCBlocks=0;         //from the next loop iteration we can wait for other data
    }
}

