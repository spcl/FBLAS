/*
    Systolic implementation.
    Elements (of A, B) are distributed in a single clock cycle on the entire
    row/column of the systolic array.
    We use vectorized types for B (column of the distributed array)
*/



#pragma OPENCL EXTENSION cl_intel_channels : enable


{% if routine.type_str == 'double' %}
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE 1
{% endif %}

#define BASE_TYPE {{ routine.type_str }}
#define VECT_SIZE {{ routine.vect_size }}
{% if routine.vect_size > 1 %}
#define VECT_TYPE {{ routine.type_str }}{{ routine.vect_size }}
{% else %}
#define VECT_TYPE {{ routine.type_str }}
{% endif %}

#define CTILE_ROWS {{ routine.width_y }}
#define CTILE_COLS {{ routine.width_x }}                  //we use vectorization along columns
#define MTILE {{ routine.tile_size }}

#if defined(DOUBLE)
    #define CHANNEL_UNROLL 4                //parallel reads in READ A
#else
    #define CHANNEL_UNROLL 8
#endif
#define WRITES_UNROLLED 1               // Parallel writes of VECT_TYPE result

typedef struct  {
    VECT_TYPE drain_data[CTILE_COLS];
} ctile_col_drain;                      // data type for drained results


channel BASE_TYPE {{ channels["channel_in_matrix_A"] }}[CTILE_ROWS] __attribute__((depth(32)));
channel VECT_TYPE {{ channels["channel_in_matrix_B"] }}[CTILE_COLS]  __attribute__((depth(32)));
channel ctile_col_drain {{ channels["channel_out_matrix"] }} __attribute__((depth(8)));


#define SHIFT_REG_SIZE ((MTILE/CTILE_ROWS*MTILE/(CTILE_COLS*VECT_SIZE)))    // number of results accumulated by each PE
#define DRAINING_SHIFT_REG_SIZE (SHIFT_REG_SIZE)                            //how many data element each PE drains

#define FPGA_REG2(x) __fpga_reg(__fpga_reg(x))
#define FPGA_REG(x) __fpga_reg(x)


//Processing Element
VECT_TYPE PE(uint it, uchar i, uchar j, BASE_TYPE a , VECT_TYPE b, VECT_TYPE *accum)
{
    //Returns the result computed in the previous iteration of loop k (see main kernel loop)
    VECT_TYPE prev = __fpga_reg(accum[0]);

    // reset according to the interation counter
    VECT_TYPE res = FPGA_REG((it<SHIFT_REG_SIZE)? 0:prev); //fpga_reg
    res += FPGA_REG(a)*FPGA_REG(b); //fpga_reg

    //shift accumulator
    #pragma unroll
    for (int i = 0; i < SHIFT_REG_SIZE-1; i++) {  //state size
        accum[i] = accum[i + 1];
    }
    accum[SHIFT_REG_SIZE - 1] = res;
    return prev;
}

__kernel void {{ routine.user_name }}(const unsigned int N, const unsigned int M, const unsigned int K, const BASE_TYPE alpha)
{

    const uint OuterBlocksN = 1 + (int)((N-1) / MTILE);
    const uint OuterBlocksM = 1 + (int)((M-1) / MTILE);
    const ushort InnerBlocksN = MTILE / CTILE_ROWS;
    const ushort InnerBlocksM = MTILE / (CTILE_COLS*VECT_SIZE);

    //shift registers for drining
    // - we have CTILE_COLS shift registers for draining
    // - each PE has to drain its entire internal state (SHIFT_REG_SIZE)
    // - each row will drain simultaneously its own content
    // - the +1 is for the element drained by the first PEs row
    VECT_TYPE draining[CTILE_COLS][(CTILE_ROWS-1)*DRAINING_SHIFT_REG_SIZE+1];

    //shift registers for PE internal states
    VECT_TYPE accum[CTILE_ROWS][CTILE_COLS][SHIFT_REG_SIZE];

    //feeding of A elemnts to PEs
    BASE_TYPE a_reg[CTILE_ROWS];
    //feeding of B elemts to PEs
    VECT_TYPE b_reg[CTILE_COLS];
    //buffer for reuse of A elmements
    BASE_TYPE a_reuse[CTILE_ROWS];
    //buffer for reuse of B elements (double bufferin)
    VECT_TYPE b_reuse[2][CTILE_COLS][InnerBlocksM];


    //Original nested loop structure flattened:
    // for(uint ti = 0 ... OuterBlocksN)
    //     for(uint tj = 0 ... OuterBlocksM)
    //         for(uint k = 0 ... K)
    //             for(uint tti = 0 ... InnerBlocksN)
    //                 for(uint ttj = 0 ... InnerBlocksM)
    //These nested loops have been flattened to save resources. In additions, other iterations are added in order
    //to drain the remaining results

    //Counters for original loops that have been flattened
    uint ti = 0;
    uint tj = 0;
    uint intra_mtile_it = 0;
    uint k = 0;
    ushort tti = 0;
    ushort ttj = 0;

    long it=0;                                              //Flattened loop counter
    bool drain_to_writer=false;                             //Indicates whether drained data must be sent to the writer
    long next_read_it = 0;                                  //Indicates when data must be read from A
    uint draining_cycles = K*InnerBlocksM*InnerBlocksN;     //how many iterations are needed for having something to drain
    long draining_it = draining_cycles ;                    //the iteration at which we have to start draining
    const uint iterations_for_draining =  MTILE * MTILE / (CTILE_COLS*VECT_SIZE); //how many iterations are needed for draining everything
    // number flattened loop iterations
    const long flattened_loop_limit = (long)OuterBlocksN * (long)OuterBlocksM * (long)K * (long)InnerBlocksN *(long)InnerBlocksM;

    long draining_stop = draining_it + iterations_for_draining;

    // Main computational loop
    #pragma ivdep
    for(long it = 0; it < flattened_loop_limit + iterations_for_draining; it++){

        bool read = (it == next_read_it);
        if(it == next_read_it)
            next_read_it += InnerBlocksM;

        #pragma unroll
        for(uchar i=0;i<CTILE_ROWS;i++){
            // We implement reuse over A: read data is reused for the next InnerBlocksM iterations

            BASE_TYPE a = -1;
            uchar my_i = i;

            BASE_TYPE value;
            if(read){
                value = read_channel_intel({{ channels["channel_in_matrix_A"] }}[i]);
                a_reuse[i]=value;
            }
            else
                value = a_reuse[i];
            a = alpha * value;
            my_i = FPGA_REG2(my_i);
            a = FPGA_REG2(a);
            a_reg[i]=a;
        }

        //We implement reuse over B: data arrived from the reader is reused for InnerBlocksN times (not consecutive)
        //use double buffering for re-using B elements
        bool to_use = k & 1;
         if(tti == 0 && it  < ((long)((long)K*(long)OuterBlocksM*(long)OuterBlocksN*(long)InnerBlocksM*(long)InnerBlocksN))){
             #pragma unroll
             for(uchar j=0; j<CTILE_COLS;j++){
                 b_reuse[to_use][j][ttj]=read_channel_intel({{ channels["channel_in_matrix_B"] }}[j]);
             }
         }
        #pragma unroll
         for(uchar j=0; j<CTILE_COLS;j++){
             b_reg[j]=b_reuse[to_use][j][ttj];
         }



        long draining_limit = (long)draining_it + SHIFT_REG_SIZE;
        bool drain = it >= draining_it  && it < draining_limit;


        //COMPUTE
        #pragma unroll
        for(uchar i=0;i<CTILE_ROWS;i++){

            #pragma unroll
            for(uchar j=0; j<CTILE_COLS;j++){
                VECT_TYPE res = -1;
                //compute at each iteration: sometimes it could be meaningless
                res=PE(intra_mtile_it,i,j,a_reg[i],b_reg[j], accum[i][j]);
                if(drain){
                    //drain result
                    draining[j][i*DRAINING_SHIFT_REG_SIZE]=res;
                }

                // Not clear here what is the most efficient combination of fpga_reg
                a_reg[i] = __fpga_reg(__fpga_reg(a_reg[i]));
            }
        }

        //build the drained result
        ctile_col_drain drained;

        #pragma unroll
        for(uchar j=0;j<CTILE_COLS;j++)
        {
            drained.drain_data[j]=draining[j][0];
            #pragma unroll
            for(uchar jj=0;jj<CTILE_COLS;jj++)
                drained.drain_data[jj]=__fpga_reg(__fpga_reg(drained.drain_data[jj])); //fpga_reg
        }

        //send drained results to writer
        if(drain_to_writer)
            write_channel_intel({{ channels["channel_out_matrix"] }}, drained);

        //compute if at the next iteration we will write again to writer
        drain_to_writer = it+1 >= draining_it &&  it < draining_it + iterations_for_draining -1;
        //if we finished draining, update the iteration number at which starts the next draining phase
        // draining_it = (it == draining_stop+1)? draining_it + draining_cycles : draining_it;
        // draining_stop = (it == draining_stop+1)? draining_it+iterations_for_draining: draining_stop;
        if(it>draining_stop){
            draining_it +=draining_cycles;
            draining_stop = draining_it+iterations_for_draining;
        }


        //how many element I have to drain?
        //- CTILE_COLS shift registers (each of size MTILE)
        //- at each clock iteration I drain up to CTILE_COLS element
        //- I have transient phases
        //= MTILE * MTILE/CTILE_COLS + CTILE_COLS-1 cycles are needed for draining

        //shift all the registers
        #pragma unroll
        for(uchar jj=0;jj<CTILE_COLS;jj++){

            #pragma unroll
            for(uchar i=0;i<CTILE_ROWS-1;i++)
            {
                #pragma unroll
                for(int ii=0;ii<DRAINING_SHIFT_REG_SIZE-1;ii++)
                     draining[jj][i*DRAINING_SHIFT_REG_SIZE+ii]=draining[jj][i*DRAINING_SHIFT_REG_SIZE+ii+1];

                draining[jj][i * DRAINING_SHIFT_REG_SIZE + DRAINING_SHIFT_REG_SIZE - 1] = __fpga_reg(__fpga_reg(draining[jj][i * DRAINING_SHIFT_REG_SIZE + DRAINING_SHIFT_REG_SIZE]));

            }
        }


        // ttj goes from 0 to InnerBlocksM
        ttj = ((ttj != InnerBlocksM -1) ? ttj +1 : 0);
        // tti goes from 0 to InnerBlocksN. Increases only when ttj goes to zero
        tti = ((ttj == 0)? ((tti != InnerBlocksN-1)? tti+1 : 0) : tti);

        //K, tj, ti are not used, we can avoid computing
        // k goes from 0 to K. Increases when tti goes to zero
        k = ((tti == 0 && ttj == 0) ? ((k != K-1) ? k+1 : 0 ) :k);
        // tj goes from 0 to OuterBlocksM
        tj = ((tti == 0 && ttj == 0 && k == 0) ? ((tj != OuterBlocksM-1) ? tj + 1: 0) : tj);
        // ti goes from 0 to OuterBlocksN
        ti = ((tti == 0 && ttj == 0 && k == 0 && tj == 0) ? ((ti != OuterBlocksN-1) ? ti + 1: 0) : ti);
        //intra_mtile_it goes from 0 to K*InnerBlocksN*InnerBlocksM
        intra_mtile_it = ((intra_mtile_it != K*InnerBlocksN*InnerBlocksM-1) ? intra_mtile_it +1 : 0);
    }
}
