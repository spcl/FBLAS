__kernel void generator_float_col_tile_row_matrix(int N, int M)
{
    const int BlocksN=N/TILE_N;
    const int BlocksM=M/TILE_M;
    int outer_loop_limit=(int)(TILE_N/W);


    for(int ti=0;ti<BlocksN;ti++)
    {
	for(int tj=0;tj<BlocksM;tj++)
	{
	    for(int j = 0; j < TILE_M; j++)
	    {
		for (int i =0; i<outer_loop_limit; i++)
		{
		    #pragma unroll
		    for(int ii=0;ii<W;ii++)
		    {
			float r = ti*TILE_N+i*W+ii+1;
			write_channel_intel(CHANNEL_MATRIX_A,r);
		    }
		}
	    }
	}
    }
}