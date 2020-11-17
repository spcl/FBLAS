/**

    This file contains helper for the bicg application.

    The included kernels are variation of helpers:
    - there is a reader from memory with interleaving. This is similar to
        read_matrix_4_modules
    - there is a vector reader, that receives updates of y and resubmit them to the gemv

*/



#pragma OPENCL EXTENSION cl_intel_channels : enable

#if defined(DOUBLE)
    #define TYPE_T double
    #define WIDTH 4 //do not change this
#else
    #define TYPE_T float
    #define WIDTH 64 //do not change this
#endif

#define TILE_N 8
#define TILE_M 8

channel TYPE_T channel_matrix __attribute__((depth(WIDTH)));
channel TYPE_T channel_matrix2 __attribute__((depth(WIDTH)));
channel TYPE_T channel_s __attribute__((depth(WIDTH)));
channel TYPE_T channel_s_updates __attribute__((depth(WIDTH)));

//N and M must be a multiple of 64 to enable hyper flex (read must be aligned)
__kernel void bicg_read_matrix(__global volatile  TYPE_T *restrict data0,__global volatile  TYPE_T *restrict data1,
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
                    for(int k = 0; k < WIDTH; k++){
                        write_channel_intel(channel_matrix,to_send[k]);
                        write_channel_intel(channel_matrix2,to_send[k]);
                    }
                }
            }
        }
    }
}

//this kernel handles s. At the first iteration it doesn't send anythin
//but then for each of tile of S it is first received (its updated value) and the re-sent toward gemv
//updates must be  sent N/TILE_N, where N is the size of vector r
__kernel void bicg_read_vector_s_updates(__global volatile TYPE_T *restrict data, int N, int updates)
{
    int BlocksY=1+(int)((N-1)/TILE_M);
    int outer_loop_limit=(int)(TILE_M/WIDTH);

    //#pragma loop_coalesce 2
    for(int k=0; k<updates; k++)
    {
        //send tile by tile
        #pragma ivdep
        for(int j=0;j<BlocksY;j++)
        {
            //at the first iteration we don't do anything
            //otherwise send the current tile
  //          for(int i=0;i<outer_loop_limit;i++)
   //         {

                if(k>0)
                {
                    for(int i=0;i<outer_loop_limit;i++)
                    {
                        TYPE_T s[WIDTH];
                        //prepare data
                        #pragma unroll
                        for(int ii=0; ii<WIDTH ;ii++){
                            if(j*TILE_M+i*WIDTH+ii<N)
                                s[ii]=data[j*TILE_M+i*WIDTH+ii];
                            else
                                s[ii]=0;
                        }

                        #pragma unroll
                        for(int ii=0; ii<WIDTH ;ii++){
                            write_channel_intel(channel_s,s[ii]);
                        }
                    }
                }

                //if this is not the last iteration
                //get back the new Y and overwrite it
                if(k<updates-1)
                {
                    for(int i=0;i<outer_loop_limit;i++)
                    {
                        TYPE_T recv[WIDTH];
                        #pragma unroll
                        for(int ii=0; ii<WIDTH ;ii++)
                        {
                            recv[ii]=read_channel_intel(channel_s_updates);
                            if(j*TILE_M+i*WIDTH+ii<N)
                                data[j*TILE_M+i*WIDTH+ii]=recv[ii];
                        }
                    }
                }

#if 0

                if(k>0)
                {
                    TYPE_T s[WIDTH];
                    //prepare data
                    #pragma unroll
                    for(int ii=0; ii<WIDTH ;ii++){
                        if(j*TILE_M+i*WIDTH+ii<N)
                            s[ii]=data[j*TILE_M+i*WIDTH+ii];
                        else
                            s[ii]=0;
                    }

                    #pragma unroll
                    for(int ii=0; ii<WIDTH ;ii++){
                        write_channel_intel(channel_s,s[ii]);
                    }
                }
                //if this is not the last iteration
                //get back the new Y and overwrite it
                if(k<updates-1)
                {
                    //for(int i=0;i<outer_loop_limit;i++)
                   // {
                        TYPE_T recv[WIDTH];
                        #pragma unroll
                        for(int ii=0; ii<WIDTH ;ii++)
                        {
                            recv[ii]=read_channel_intel(channel_s_updates);
                            if(j*TILE_M+i*WIDTH+ii<N)
                                data[j*TILE_M+i*WIDTH+ii]=recv[ii];
                        }
                    //}
                }
#endif
         //   }


        }
//	    printf("Generator Y finito ricevuto update %d\n",k+1);
    }
}