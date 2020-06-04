/*
    This file contains helper for the gemver application.

    The helper is a classical matrix writer (tiles by column) but in this case
    the matrix also forwared to another channel (in gemer the transposed gemv)

*/



#pragma OPENCL EXTENSION cl_intel_channels : enable

#if defined(DOUBLE)
    #define TYPE_T double
    #define WIDTH 16 //do not change this
#else
    #define TYPE_T float
    #define WIDTH 32 //do not change this
#endif

#define TILE_N 2048
#define TILE_M 2048

channel TYPE_T channel_matrix_B __attribute__((depth(WIDTH)));
channel TYPE_T channel_matrix_B_trans __attribute__((depth(WIDTH)));

__kernel void write_and_forward_matrix_B(__global volatile TYPE_T *restrict matrix, int N, int M, unsigned int lda)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    int outer_loop_limit=(int)(TILE_M/WIDTH);


    for(int tj=0;tj<BlocksM;tj++)
    {
        for(int ti=0;ti<BlocksN;ti++)
        {
            for(int i = 0; i < TILE_N; i++)
            {
                for(int j=0;j<outer_loop_limit;j++)
                {
                    TYPE_T r[WIDTH];
                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                        r[jj] = read_channel_intel(channel_matrix_B);

                    //send to the next one
                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                        write_channel_intel(channel_matrix_B_trans,r[jj]);


                    //write
                    #pragma unroll
                    for(int jj= 0; jj < WIDTH; jj++)
                        if((ti*TILE_N+i)<N  && tj*TILE_N+j*WIDTH+jj< M) //skip padding data
                            matrix[(ti*TILE_N+i)*lda+(tj*TILE_M+j*WIDTH+jj)]=r[jj];
                }
            }
        }
    }
}