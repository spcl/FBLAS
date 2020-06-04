/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Contains definition of data types and constants used by code generators and API
*/

#ifndef COMMONS_HPP
#define COMMONS_HPP


typedef enum FblasOrder     {FBLAS_ROW_MAJOR=101, FBLAS_COL_MAJOR=102, FBLAS_O_UNDEF = 109} FblasOrder;
typedef enum FblasTranspose {FBLAS_NO_TRANSPOSED=111, FBLAS_TRANSPOSED=112, FBLAS_T_UNDEF = 119} FblasTranspose;
typedef enum FblasUpLo      {FBLAS_UPPER=121, FBLAS_LOWER=122, FBLAS_UL_UNDEF = 129} FblasUpLo;
typedef enum FblasSide      {FBLAS_LEFT=131, FBLAS_RIGHT=132, FBLAS_SIDE_UNDEF = 139} FblasSide;
typedef enum FblasDiag      {FBLAS_UNIT=141, FBLAS_NO_UNIT=142, FBLAS_DIAG_UNDEF = 149} FblasDiag;

typedef enum FblasArchitecture{FBLAS_STRATIX_10=191, FBLAS_ARRIA_10=192} FBlasArchiteture; //architectures




/*
 *  Constants used in the JSON produced for runtime
 */

//Properties fields

const char* const k_json_field_width="width";       //used for Level 1/2
const char* const k_json_field_width_x="width x";   //these two are used for Level 3
const char* const k_json_field_width_y="width y";
const char* const k_json_field_blas_name="blas_name";
const char* const k_json_field_user_name="user_name";
const char* const k_json_field_precision="precision";
const char* const k_json_field_order="order";
const char* const k_json_field_transA="transa";
const char* const k_json_field_transB="transb";
const char* const k_json_field_uplo="uplo";
const char* const k_json_field_side="side";
const char* const k_json_field_tile_n_size="tile N size";
const char* const k_json_field_tile_m_size="tile M size";
const char* const k_json_field_mtile_size="tile size";
const char* const k_json_field_incx="incx";
const char* const k_json_field_incy="incy";
const char* const k_json_field_lda="lda";
const char* const k_json_field_ldb="ldb";
const char* const k_json_field_systolic="systolic";



//Helper fields
const char* const k_json_field_read_vector_x="read_vector_x";
const char* const k_json_field_read_vector_y="read_vector_y";
const char* const k_json_field_read_vector_x_trans="read_vector_x_trans";
const char* const k_json_field_read_vector_y_trans="read_vector_y_trans";
const char* const k_json_field_read_vector_x_trsv="read_vector_x_trsv";
const char* const k_json_field_write_scalar="write_scalar";
const char* const k_json_field_write_vector="write_vector";
const char* const k_json_field_read_matrix_A="read_matrix_A";
const char* const k_json_field_read_matrix_A2="read_matrix_A2";
const char* const k_json_field_read_matrix_B="read_matrix_B";
const char* const k_json_field_read_matrix_B2="read_matrix_B2";
const char* const k_json_field_write_matrix="write_matrix";




#endif // COMMONS_HPP
