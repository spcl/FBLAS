/**
    This file contains the two helpers needed to run the dot program.
    They are in charge of reading the two input vector (x,y) from DRAM.

    The two readers are similar to the standard vector readers.
    To exploit full memory bandwidth, we have to manually interleave between
    DRAM modules (Intel Compiler doesn't do this automatically).

    Since these will be used by both single and double precision, the TYPE_T macro
    defines the type of data (float/double).


**/

#pragma OPENCL EXTENSION cl_intel_channels : enable

#if defined(DOUBLE)
    #define TYPE_T double
    #define WIDTH 16 //do not change this
#else
    #define TYPE_T float
    #define WIDTH 32 //do not change this
#endif
#define INCX 1
#define INCY 1

channel TYPE_T channel_x __attribute__((depth(WIDTH)));
channel TYPE_T channel_y __attribute__((depth(WIDTH)));

__kernel void read_vector_x(__global TYPE_T *restrict data0,__global TYPE_T *restrict data1, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;

    #pragma loop_coalesce
    for(int t=0; t < repetitions;t++)
    {
        //compute the starting index
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

        for(int i=0;i<outer_loop_limit;i++)
        {
            TYPE_T x[WIDTH];
            //prepare data
            #pragma unroll
            for(int k=0;k<WIDTH/2;k++)
            {
                if(i*WIDTH/2+k<N)
                   x[k]=data0[offset+(k*INCX)];
                else
                    x[k]=0;
            }

            #pragma unroll
            for(int k=0;k<WIDTH/2;k++)
            {
                if(i*WIDTH/2+k<N)
                   x[k+WIDTH/2]=data1[offset+(k*INCX)];
                else
                    x[k+WIDTH/2]=0;
            }

            offset+=WIDTH/2*INCX;

            //send data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
                write_channel_intel(channel_x,x[k]);
        }
    }
}


__kernel void read_vector_y(__global TYPE_T *restrict data0,__global TYPE_T *restrict data1, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;

    #pragma loop_coalesce
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));
        for(int i=0;i<outer_loop_limit;i++)
        {
            TYPE_T y[WIDTH];
            //prepare data
            #pragma unroll
            for(int k=0;k<WIDTH/2;k++)
            {
                if(i*WIDTH/2+k<N)
                    y[k]=data0[offset+(k*INCY)];
                else
                    y[k]=0;
            }
            #pragma unroll
            for(int k=0;k<WIDTH/2;k++)
            {
                if(i*WIDTH/2+k<N)
                    y[k+WIDTH/2]=data1[offset+(k*INCY)];
                else
                    y[k+WIDTH/2]=0;
            }
            offset+=WIDTH/2*INCY;

            //send data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
                write_channel_intel(channel_y,y[k]);
        }
    }
}