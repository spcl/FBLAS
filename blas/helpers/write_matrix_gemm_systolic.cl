/*
*/
__kernel void WRITE_MATRIX(__global volatile TYPE_T * restrict C, const float beta,const unsigned int N, const unsigned int M, const unsigned int ldc)
{
    //this kernel will receive the data for C in order
    const int OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const int OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const int InnerBlocksN = MTILE / CTILE_ROWS;
    const int InnerBlocksM = MTILE / CTILE_COLS;

    int received=0;
    //for each outer tile of C, receive and accumulate
    #pragma loop_coalesce
    #pragma ivdep array(C)
    for(int ti=0;ti<OuterBlocksN;ti++)
    {

        #pragma ivdep array(C)
        for(int tj=0;tj<OuterBlocksM;tj++)
        {
            //read and save
            #pragma ivdep array(C)
            for(int iii=0;iii<CTILE_ROWS;iii++)
            {
                #pragma ivdep array(C)
                for(int ii=0;ii<MTILE/CTILE_ROWS;ii++)
                {

                    #pragma ivdep array(C)
                    for(int jj=0;jj<MTILE/CTILE_COLS;jj++)
                    {
                        #if defined(DOUBLE_PRECISION)
                        ctile_col_double res;
                        #else
                        ctile_col_float res;
                        #endif
                        res=read_channel_intel(CHANNEL_MATRIX_OUT);
                        #pragma unroll
                        for(int jjj=0;jjj<CTILE_COLS;jjj++)
                        {
                             int ind_i=ti*MTILE+ii*CTILE_ROWS+iii;
                             int ind_j=tj*MTILE+jj*CTILE_COLS+jjj;
                             //float c =
                             if(ind_i < N && ind_j<M)
                                 C[ind_i*ldc+ind_j]=beta*C[ind_i*ldc+ind_j]+res.drain_data[jjj];
                        }
                     }
                 }
            }

        }
    }
}
