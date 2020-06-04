/**
    This version solves the case: side left, A lower, non transposed.
    No distinction is made between unit/non unit cases

    Fully unrolled

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

#define KERNEL_NAME trsm_v1
#define __STRATIX_10__

#define TYPE_T double

#define TRSM_N 4    //# rows matrix B
#define TRSM_M 4    //# cols matrix B

#define LDA TRSM_N
#define LDB TRSM_M

channel TYPE_T trsm_A_in __attribute__((depth(TRSM_N)));
channel TYPE_T trsm_B_in __attribute__((depth(TRSM_N)));
channel TYPE_T trsm_B_out __attribute__((depth(TRSM_N)));


__kernel void strsm(const int N, const int M, const TYPE_T alpha, const int ntrsm)
{
    for(int nt = 0; nt < ntrsm; nt++){

        TYPE_T localA[TRSM_N][TRSM_M];
        #pragma unroll
        for(int i=0; i<TRSM_N;i ++)
            #pragma unroll
            for(int j=0; j< TRSM_N;j++)
                localA[i][j]=read_channel_intel(trsm_A_in);

        TYPE_T localB[TRSM_N][TRSM_M];
        #pragma unroll
        for(int i=0; i<TRSM_N;i ++)
            #pragma unroll
            for(int j=0; j< TRSM_N;j++)
                localB[i][j]=read_channel_intel(trsm_B_in);


        #pragma unroll
        for(int i=0;i<TRSM_N;i++)
        {
            const TYPE_T aii=localA[i][i];



            #pragma unroll
            for(int j=0;j<TRSM_M;j++)
            {
                if(j < M)
                {
                    const TYPE_T prev=(i==0)?alpha*localB[i][j]:localB[i][j];
                    localB[i][j]=prev/aii;

                }
            }



            #pragma unroll  //unroll everything to favor the compiler
            for(int k=0;k<TRSM_N;k++)
            {
                if(k>=i+1){
                    TYPE_T aki=localA[k][i];
                    // const int off_idx=k*ldb+tj*TILE_M+jj*W;

                    #pragma unroll
                    for(int j=0;j<TRSM_M;j++)
                    {

                        const TYPE_T prev=(i==0)?alpha*localB[k][j]:localB[k][j];
                        localB[k][j] = prev - aki * localB[i][j];

                    }

                }
            }

        }
        #pragma unroll
        for(int i=0; i<TRSM_N;i ++)
            #pragma unroll
            for(int j=0; j< TRSM_N;j++)
                write_channel_intel(trsm_B_out,localB[i][j]);
    }
}


__kernel void read_matrix_A(__global volatile TYPE_T * restrict A, const int ntrsm)
{
    int offset=0;
    for(int ng = 0; ng < ntrsm; ng ++){

        #pragma unroll
        for(int i=0;i<TRSM_N;i++)
        {
            #pragma unroll
            for(int j=0;j<TRSM_N;j++)
            {
                #if !defined(NO_MEMORY)
                write_channel_intel(trsm_A_in,A[offset + i*LDA+j]);
                #else
                write_channel_intel(CHANNEL_MATRIX_A,i*LDA+j);
                #endif
            }
        }
        offset +=TRSM_N*TRSM_N;
    }
}


__kernel void read_matrix_B(__global volatile TYPE_T * restrict B, const int ntrsm)
{
    int offset=0;
    for(int ng = 0; ng < ntrsm; ng ++){

        #pragma unroll
        for(int i=0;i<TRSM_N;i++)
        {
            #pragma unroll
            for(int j=0;j<TRSM_M;j++)
            {
                #if !defined(NO_MEMORY)
                write_channel_intel(trsm_B_in,B[offset + i*LDB+j]);
                #else
                write_channel_intel(trsm_B_in,i*LDA+j);
                #endif
            }
        }
        offset +=TRSM_N*TRSM_M;
    }
}


__kernel void write_matrix_B(__global volatile TYPE_T * restrict B, const int ntrsm)
{
    int offset=0;
    for(int ng = 0; ng < ntrsm; ng ++){

        #pragma unroll
        for(int i=0;i<TRSM_N;i++)
        {
            #pragma unroll
            for(int j=0;j<TRSM_M;j++)
            {
                #if !defined(NO_MEMORY)

                B[offset + i*LDB+j] = read_channel_intel(trsm_B_out);
                #else
                read_channel_intel(trsm_B_out);
                #endif
            }
        }
        offset +=TRSM_N*TRSM_M;
    }
}
