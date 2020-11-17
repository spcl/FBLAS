/**
    This file contains the helper for reading the matrix needed to run the gemv program.

    The reader is similar to the standard matrix reader (tiles by row, row streamed).
    To exploit full memory bandwidth, we have to manually interleave between
    DRAM modules (Intel Compiler doesn't do this automatically).

    Since these will be used by both single and double precision, the TYPE_T macro
    defines the type of data (float/double).


**/

#pragma OPENCL EXTENSION cl_intel_channels : enable

#if defined(DOUBLE)
    #define TYPE_T double
    #define WIDTH 32 //do not change this
#else
    #define TYPE_T float
    #define WIDTH 64 //do not change this
#endif

#define TILE_N 2048
#define TILE_M 2048

channel TYPE_T channel_matrix __attribute__((depth(WIDTH)));

//N and M must be a multiple of 64 to enable hyper flex (read must be aligned)
__kernel void sgemv_read_matrix(__global volatile  TYPE_T *restrict data0,__global volatile  TYPE_T *restrict data1,
                            __global volatile  TYPE_T *restrict data2,__global volatile  TYPE_T *restrict data3, int N, int M, unsigned int lda)
{
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int loop_it=((int)(TILE_M))/WIDTH;   //W must be a divisor of TILE
    const int multiply_width=1+(int)((lda-1)/WIDTH); //lda must be a multiple of width, otherwise inefficient hw is generated for the load

    TYPE_T to_send[WIDTH];
     #pragma loop_coalesce
    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i=0;i<TILE_N;i++)
            {
                for(int j=0;j<loop_it;j++)
                {
                    const int row_idx=ti*TILE_N+i;
                    //load from memory

                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k]=data0[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];

                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k+WIDTH/4]=data1[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];
                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k+WIDTH/2]=data2[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];
                    #pragma unroll
                    for(int k=0;k<WIDTH/4;k++)
                            to_send[k+3*WIDTH/4]=data3[row_idx*WIDTH/4*multiply_width+tj*TILE_M/4+j*WIDTH/4+k];

                    #pragma unroll
                    for(int k = 0; k < WIDTH; k++)
                        write_channel_intel(channel_matrix,to_send[k]);
                }
            }
        }
    }
}