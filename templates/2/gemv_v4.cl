/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GEMV_V4  performs one of the matrix-vector operations

    -   y := alpha*A*x + beta*y,
            where the NxM matrix A is received in tiles streamed by column, where each
            tile is ColStreamed. x is an M-elements vector, while y
            is an N-element vector.
            y must be replayed entirely M/TILE_M times.

    -   or  y := alpha*A**T*x + beta*y,
            where the NxM matrix A is received in tiles streamed by rows,
            each tile is Row Streamed. x is an N-element vector, while y
            is an M-element vector
            y must be replayed entirely N/TILE_N times

    Data is received from three different channels ({{ channels["channel_in_vector_x"] }}, {{ channels["channel_in_vector_y"] }}
    and CHANNEL_MATRIX A). Input data must be padded with zeros according to
    the reference tile sizes (TILE_N and TILE_M).

	In this version, vector y must be updated and re-injected (into {{ channels["channel_in_vector_y"] }}) multiple
	times. The elements of vector y that must be updated are sent to {{ channels["channel_out_vector_y_updates"] }}

    Result is streamed in the output channel, one element at a time as soon as it is available.

    Check the kernel documentation for further information

    NOTE: deserves further optimizations
*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
{% endif %}


channel {{ routine.type_str }} {{ channels["channel_in_vector_x"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_vector_y"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector_y_updates"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_in_matrix_A"] }} __attribute__((depth({{ routine.width }})));
channel {{ routine.type_str }} {{ channels["channel_out_vector"] }} __attribute__((depth(1)));



/**
    This version is meant for the following cases:
    - A is rowStreamed and Transposed. A is received by tiles, in row ordering. For this case,
	- vector x is composed by N elements, y is composed by M elements
	- blocks of x are composed by Tile_N elements, they will be reused
	- blocks of y are composed by Tile_M elements and they will be not reused
	- y must be re-sent entirely N/TILE_N times. For each Row tile it is entirely updated

    - A is columnStreamed and Not Transposed. A is received by tiles in col ordering:
	- vector x is composed by M elements, vector y is composed by N elements
	- block of x are composed by TILE_M elements,they will be reused
	- block of y are composed by TILE_N elements and they will be not reused
	- y must be re-sent entirely M/TILE_M times. For each Column tile it is entirely updated

*/

__kernel void {{ routine.user_name }}(int row_streamed, const int N, const int M, const {{ routine.type_str }} alpha, const {{ routine.type_str }} beta)
{
    __constant uint WIDTH = {{ routine.width }};
    __constant uint TILE_N = {{ routine.tile_n_size }};
    __constant uint TILE_M = {{ routine.tile_m_size }};
    {% if routine.tile_n_size > routine.tile_m_size %}
    __constant uint MAX_TILE_SIZE = {{ routine.tile_n_size }};
    {% else %}
    __constant uint MAX_TILE_SIZE = {{ routine.tile_m_size }};
    {% endif %}

    int len_x,tile_x;
    int len_y,tile_y;
    int BlocksY, BlocksX;
    //chose the loop limits
    if(row_streamed==1)
    {
		//in this case A is transposed
		len_x = N;
		len_y = M;
		tile_x=TILE_N;
		tile_y=TILE_M;
		BlocksY=M/TILE_M;
		BlocksX=N/TILE_N;
		BlocksY=1+(int)((M-1)/TILE_M);
        BlocksX=1+(int)((N-1)/TILE_N);
    }
    else
    {
		len_x = M;
		len_y = N;
		tile_x=TILE_M;
		tile_y=TILE_N;
	  	BlocksY=1+(int)((N-1)/TILE_N); //ceiling
        BlocksX=1+(int)((M-1)/TILE_M);

    }

    const int computing_outer_loop_limit=(int)(tile_y/WIDTH);
    const int reading_y_outer_loop_limit=(int)(tile_y/WIDTH);
    const int reading_x_outer_loop_limit=(int)(tile_x/WIDTH);

    //Please note: the order in which tiles arrive, will determine the computation
    //(i.e. do not assume that you will receive the tiles one row after the other...maybe they can arrive column by column)
    // - for the case A row streamed and transposed they will arrive row by row
    //#pragma ivdep
    {{ routine.type_str }} local_y[MAX_TILE_SIZE];
    for(int ti=0;ti<BlocksX;ti++)
    {
        //load x, it will be reused for the entire tile row(or column)
                {{ routine.type_str }} local_x[MAX_TILE_SIZE];
                for(int i=0;i<reading_x_outer_loop_limit;i++)
                {
                #pragma unroll
                for(int j=0;j<WIDTH;j++)
                                local_x[i*WIDTH+j]=read_channel_intel({{ channels["channel_in_vector_x"] }});
                }


                for(int tj=0;tj<BlocksY;tj++)
                {
                //	printf("GEMVT blocco %d di %d\n",tj,BlocksY);
                    //load the block of y
                    for(int i=0;i<reading_y_outer_loop_limit;i++)
                    {
                        if(beta == 0  && ti == 0)
                        {
                                #pragma unroll
                        for(int j=0;j<WIDTH;j++)
                        local_y[i*WIDTH+j] = 0;
                        }
                        else
                        {
                                const {{ routine.type_str }} multiplier=(ti==0)? beta: 1;
                                #pragma unroll
                        for(int j=0;j<WIDTH;j++)
                        local_y[i*WIDTH+j]=multiplier*read_channel_intel({{ channels["channel_in_vector_y"] }});
                        }

                    }


                    #pragma ivdep
                    for(int i=0;i<tile_x;i++)
                    {
                        //for each row in this tile, we will update each element of the block of y
                        {{ routine.type_str }} temp=alpha*local_x[i];


                        //here we read one row/column of A and multiply it for the same value of x
                        for(int jj=0;jj<computing_outer_loop_limit;jj++)
                        {
                            //receive elemnts of a: decoupling this form the computation loop
                            //maybe usefule in case the sender of A does not perform unrolled writes into the channel
                            {{ routine.type_str }} local_A[WIDTH];
                            #pragma unroll
                            for(int j=0;j<WIDTH;j++)
                                    local_A[j]=read_channel_intel({{ channels["channel_in_matrix_A"] }});

                            //updates all y
                            #pragma unroll
                            for(int j=0;j<WIDTH;j++)
                                        local_y[jj*WIDTH+j]+=local_A[j]*temp;



                        }
                                //Here we can not send y before we finish this column of tiles (if we are in the rowStreamed case)
                                //this because each line of A will contributes to every element in the block of y


                    }
                    // send the updated y for reply
                    //send it in output
                    if(ti==BlocksX-1)
                    {
                                for(int i=0;i<computing_outer_loop_limit;i++)
                                        #pragma unroll
                                        for(int ii=0;ii<WIDTH;ii++)
                                        write_channel_intel({{ channels["channel_out_vector"] }},local_y[i*WIDTH+ii]);

                    }
                    else
                    {
                                for(int i=0;i<computing_outer_loop_limit;i++)
                                        #pragma unroll
                                        for(int ii=0;ii<WIDTH;ii++)
                                            write_channel_intel({{ channels["channel_out_vector_y_updates"] }},local_y[i*WIDTH+ii]);
                    }
                }
    }
}
