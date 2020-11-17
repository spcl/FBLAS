/**
    Write two vectors of type {{ helper.type_str }} into memory.
    The vector elements are read from channels
    {{ channels["channel_in_vector_x"] }} and {{ channels["channel_in_vector_y"] }}.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCX and INCY represent the access strides.

    W reads are performed simultaneously.
    Data arrives padded at pad_size.
    Padding data (if present) is discarded.

    This helper can be used with ROT and ROTM routines

*/
__kernel void {{ helper_name }}(__global {{ helper.type_str }} *restrict out_x, __global {{ helper.type_str }} *restrict out_y,  unsigned int N,unsigned int pad_size)
{

    __constant uint WIDTH = {{ helper.width }};
    __constant int INCX = {{ helper.incx }};
    __constant int INCY = {{ helper.incy }};
    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;

    {{ helper.type_str }} recv_x[WIDTH], recv_y[WIDTH];
    int offset_x=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));
    int offset_y=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));

    for(int i=0;i<outer_loop_limit;i++)
    {
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
        {
            recv_x[j]=read_channel_intel({{ channels["channel_in_vector_x"] }});
            recv_y[j]=read_channel_intel({{ channels["channel_in_vector_y"] }});

            if(i*WIDTH+j<N)
            {
                out_x[offset_x+(j*INCX)]=recv_x[j];
                out_y[offset_y+(j*INCY)]=recv_y[j];
            }

        }

        offset_x+=WIDTH*INCX;
        offset_y+=WIDTH*INCY;
    }
}