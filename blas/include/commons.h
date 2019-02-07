/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.
*/

#ifndef COMMONS_CL_HPP
#define COMMONS_CL_HPP


/**
 * A set of commons definitions that has to be included in BLAS
 * routine kernels
 *
 * Must be included after the definition of the DOUBLE_PRECISION macro (if needed)
 */


#ifdef DOUBLE_PRECISION
#define TYPE_T double
#else
#define TYPE_T float	    //type of data: float if DOUBLE_PRECISION is undefined, double otherwise
#endif


#ifdef DOUBLE_PRECISION
//enable double precision support
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifdef __STRATIX_10__
#define DOUBLE_ADD_LATENCY 28	//double add latency for Stratix10
#endif

#ifdef __ARRIA_10__
#define DOUBLE_ADD_LATENCY 12	//double add latency for Arria 10
#endif

#define SHIFT_REG DOUBLE_ADD_LATENCY+6 //Shift register dimension for double precision operations (additional elements to avoid Fmax problems)
#endif
#endif
