/**

   Fully unrolled version of GEMM

*/



#pragma OPENCL EXTENSION cl_intel_channels : enable

#define GEMM_N 4
#define GEMM_M 4
#define GEMM_K 4

#define LDA GEMM_K
#define LDB GEMM_M
#define LDC GEMM_M

#define CHANNEL_MATRIX_A channel_in_matrix_A_0
#define CHANNEL_MATRIX_B channel_in_matrix_B_0
#define CHANNEL_MATRIX_OUT channel_out_matrix_0
#define READ_MATRIX_A kernel_read_matrix_A_0
#define READ_MATRIX_B kernel_read_matrix_B_0
#define WRITE_MATRIX kernel_write_matrix_0
#define __STRATIX_10__

#define TYPE_T double

channel TYPE_T CHANNEL_MATRIX_A __attribute__((depth(GEMM_N)));
channel TYPE_T CHANNEL_MATRIX_B __attribute__((depth(GEMM_M)));
channel TYPE_T CHANNEL_MATRIX_C_IN __attribute__((depth(GEMM_M)));
channel TYPE_T CHANNEL_MATRIX_OUT __attribute__((depth(GEMM_M)));

//#define NO_MEMORY 1

/**

    SPECIALIZED VERSION:
    - N, M and K are known at compile time
    - There is no double level of tiling (essentially MTILE=CTILE)
    - everything is unrolled so that the result is produced in one clock cycle
    - A is Non Transposed, B is NonTransposed

*/


__kernel void sgemm(const int N, const int M, const int K, const TYPE_T alpha, const TYPE_T beta, const int ngemm)
{

    for(int ng = 0; ng < ngemm; ng ++){

        //this must be implemented using register
        TYPE_T localC[GEMM_N][GEMM_M];
        #pragma unroll
        for(int i=0; i<GEMM_N;i ++)
            #pragma unroll
            for(int j=0; j< GEMM_M;j++)
                localC[i][j]=beta*read_channel_intel(CHANNEL_MATRIX_C_IN);


        //GEMM: everything is unrolled

        #pragma unroll
        for(int k=0;k<GEMM_K;k++)
        {
            TYPE_T localA[GEMM_N];
            TYPE_T localB[GEMM_M];


            #pragma unroll
            for(int i=0;i<GEMM_N;i++)
              localA[i]=read_channel_intel(CHANNEL_MATRIX_A);

            #pragma unroll
            for(int i=0;i<GEMM_M;i++)
                localB[i]=read_channel_intel(CHANNEL_MATRIX_B);

            //to unroll
            #pragma unroll
            for(int i=0;i<GEMM_N;i++)
            {
                TYPE_T tmpa=alpha*localA[i];
                #pragma unroll
                for(int j=0; j<GEMM_M;j++)
                {

                    localC[i][j]+=tmpa*localB[j];
                }
            }

        }

        //prevent unrolls on this, for the sake of saving BRAMs

        #pragma unroll
        for(int iii=0;iii<GEMM_N;iii++)
            #pragma unroll
            for(int jjj=0;jjj<GEMM_M;jjj++)
            {
                    int ind_i=iii;
                    int ind_j=jjj;
                    write_channel_intel(CHANNEL_MATRIX_OUT,localC[iii][jjj]);
             }
    }


}



__kernel void read_matrix_A(__global volatile TYPE_T * restrict A, const int ngemm)
{
    int offset=0;
    for(int ng = 0; ng < ngemm; ng ++){

        #pragma unroll
        for(int k=0;k<GEMM_K;k++)
        {
            #pragma unroll
            for(int i=0;i<GEMM_N;i++)
            {
                #if !defined(NO_MEMORY)
                write_channel_intel(CHANNEL_MATRIX_A,A[offset + i*LDA+k]);
                #else
                write_channel_intel(CHANNEL_MATRIX_A,i*LDA+k);
                #endif
            }
        }
        offset +=GEMM_N*GEMM_M;
    }
}


__kernel void read_matrix_B(__global volatile TYPE_T * restrict B, const int ngemm)
{
    uint offset = 0;
    for(int ng = 0; ng < ngemm; ng ++){

        #pragma unroll
        for(int j=0;j<GEMM_K;j++)
        {
            #pragma unroll
            for(int k=0;k<GEMM_M;k++)
            {
                #if !defined(NO_MEMORY)
                write_channel_intel(CHANNEL_MATRIX_B,B[offset + j*LDB + k]);
                #else
                write_channel_intel(CHANNEL_MATRIX_B,j*LDB+k);
                #endif

            }

        }
        offset +=GEMM_N*GEMM_M;
    }
}

__kernel void read_matrix_C(__global volatile TYPE_T * restrict C, const int ngemm)
{

    uint offset = 0;
    for(int ng = 0; ng < ngemm; ng ++){

        #pragma unroll
        for(int i=0;i<GEMM_N;i++)
        {
            #pragma unroll
            for(int j=0;j<GEMM_M;j++)
            {
                #if !defined(NO_MEMORY)
                write_channel_intel(CHANNEL_MATRIX_C_IN,C[offset + i*LDC + j]);
                #else
                write_channel_intel(CHANNEL_MATRIX_C_IN,1.0f);
                #endif
            }

        }
        offset +=GEMM_N*GEMM_M;
    }
}

__kernel void write_matrix_C(__global volatile TYPE_T * restrict C, const int ngemm)
{
    uint offset = 0;
    for(int ng = 0; ng < ngemm; ng ++){
        TYPE_T c;
        #pragma unroll
        for(int i=0;i<GEMM_N;i++)
            #pragma unroll
            for(int j=0;j<GEMM_M;j++)
            {
                 c = read_channel_intel(CHANNEL_MATRIX_OUT);
                #if !defined(NO_MEMORY)
                C[offset +i*LDC+j]=c;
                #endif

            }
        // #if !defined(NO_MEMORY)
        // C[offset]=c;
        // #endif
        offset +=GEMM_N*GEMM_M;
    }


}

