/**
    Write two vectors of type TYPE_T into memory.
    The vector elements are read from channels
    CHANNEL_VECTOR_OUT_X and CHANNEL_VECTOR_OUT_Y.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCX and INCY represent the access strides.

    W reads are performed simultaneously.
    Data arrives padded at pad_size.
    Padding data (if present) is discarded.

    This helper can be useed with ROT and ROTM routines

*/
__kernel void WRITE_VECTORS(__global TYPE_T *restrict out_x, __global TYPE_T *restrict out_y,  unsigned int N,unsigned int pad_size)
{
	const unsigned int ratio=pad_size/W;
	const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
	const unsigned int outer_loop_limit=padding_loop_limit*ratio;

	TYPE_T recv_x[W], recv_y[W];
	int offset_x=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));
	int offset_y=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));

	for(int i=0;i<outer_loop_limit;i++)
	{
	  	#pragma unroll
		for(int j=0;j<W;j++)
		{
			recv_x[j]=read_channel_intel(CHANNEL_VECTOR_OUT_X);
			recv_y[j]=read_channel_intel(CHANNEL_VECTOR_OUT_Y);

			if(i*W+j<N)
			{
                out_x[offset_x+(j*INCX)]=recv_x[j];
                out_y[offset_y+(j*INCY)]=recv_y[j];
			}

		}

		offset_x+=W*INCX;
		offset_y+=W*INCY;
	}
}