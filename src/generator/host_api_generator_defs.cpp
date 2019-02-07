/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Contains definition of data types and constants used by code generators and API
*/

#include "../../include/generator/fblas_generator.hpp"
#include "../../include/commons.hpp"

const std::string FBlasGenerator::k_skeleton_folder_=std::string(BASE_FOLDER)+std::string("/blas/");
const std::string FBlasGenerator::k_double_precision_define_="#define DOUBLE_PRECISION";
const std::string FBlasGenerator::k_width_define_="#define W ";
const std::string FBlasGenerator::k_ctile_rows_define_="#define CTILE_ROWS ";
const std::string FBlasGenerator::k_ctile_cols_define_="#define CTILE_COLS ";
const std::string FBlasGenerator::k_tile_n_size_define_="#define TILE_N ";
const std::string FBlasGenerator::k_tile_m_size_define_="#define TILE_M ";
const std::string FBlasGenerator::k_mtile_size_define_="#define MTILE ";
const std::string FBlasGenerator::k_kernel_name_define_ = "#define KERNEL_NAME ";
const std::string FBlasGenerator::k_channel_x_define_="#define CHANNEL_VECTOR_X ";
const std::string FBlasGenerator::k_channel_y_define_="#define CHANNEL_VECTOR_Y ";
const std::string FBlasGenerator::k_channel_x_trans_define_="#define CHANNEL_VECTOR_X_TRANS ";
const std::string FBlasGenerator::k_channel_y_trans_define_="#define CHANNEL_VECTOR_Y_TRANS ";
const std::string FBlasGenerator::k_channel_x_trsv_define_="#define CHANNEL_VECTOR_X ";
const std::string FBlasGenerator::k_channel_matrix_A_define_="#define CHANNEL_MATRIX_A ";
const std::string FBlasGenerator::k_channel_matrix_A2_define_="#define CHANNEL_MATRIX_A2 ";
const std::string FBlasGenerator::k_channel_matrix_B_define_="#define CHANNEL_MATRIX_B ";
const std::string FBlasGenerator::k_channel_matrix_B2_define_="#define CHANNEL_MATRIX_B2 ";
const std::string FBlasGenerator::k_channel_scalar_out_define_="#define CHANNEL_OUT ";
const std::string FBlasGenerator::k_channel_vector_out_define_="#define CHANNEL_VECTOR_OUT ";
const std::string FBlasGenerator::k_channel_vector_out_x_define_="#define CHANNEL_VECTOR_OUT_X ";
const std::string FBlasGenerator::k_channel_vector_out_y_define_="#define CHANNEL_VECTOR_OUT_Y ";
const std::string FBlasGenerator::k_channel_matrix_out_define_="#define CHANNEL_MATRIX_OUT ";

const std::string FBlasGenerator::k_stratix_10_platform_define_="#define __STRATIX_10__";
const std::string FBlasGenerator::k_arria_10_platform_define_="#define __ARRIA_10__";

const std::string FBlasGenerator::k_commons_define_="#include <commons.h>";

//kernel names defines
const std::string FBlasGenerator::k_read_vector_x_name_define_="#define READ_VECTOR_X ";
const std::string FBlasGenerator::k_read_vector_y_name_define_="#define READ_VECTOR_Y ";
const std::string FBlasGenerator::k_read_vector_x_trans_name_define_="#define READ_VECTOR_X_TRANS ";
const std::string FBlasGenerator::k_read_vector_y_trans_name_define_="#define READ_VECTOR_Y_TRANS ";
const std::string FBlasGenerator::k_read_vector_x_trsv_name_define_="#define READ_VECTOR_X_TRSV ";
const std::string FBlasGenerator::k_read_matrix_A_name_define_="#define READ_MATRIX_A ";
const std::string FBlasGenerator::k_read_matrix_A2_name_define_="#define READ_MATRIX_A2 ";
const std::string FBlasGenerator::k_read_matrix_B_name_define_="#define READ_MATRIX_B ";
const std::string FBlasGenerator::k_read_matrix_B2_name_define_="#define READ_MATRIX_B2 ";
const std::string FBlasGenerator::k_write_scalar_name_define_="#define WRITE_SCALAR ";
const std::string FBlasGenerator::k_write_vector_name_define_="#define WRITE_VECTOR ";
const std::string FBlasGenerator::k_write_vector_x_y_name_define_="#define WRITE_VECTORS ";
const std::string FBlasGenerator::k_write_matrix_name_define_="#define WRITE_MATRIX ";
const std::string FBlasGenerator::k_incx_define_="#define INCX ";
const std::string FBlasGenerator::k_incy_define_="#define INCY ";
const std::string FBlasGenerator::k_incw_define_="#define INCW ";

//helpers file names
//we need this especially for the matrices, because we have different versions of them
const std::string FBlasGenerator::k_helper_read_vector_x_="/helpers/read_vector_x.cl";
const std::string FBlasGenerator::k_helper_read_vector_y_="/helpers/read_vector_y.cl";
const std::string FBlasGenerator::k_helper_read_vector_x_trans_low_="/helpers/read_vector_x_incremental_low.cl";
const std::string FBlasGenerator::k_helper_read_vector_y_trans_low_="/helpers/read_vector_y_incremental_low.cl";
const std::string FBlasGenerator::k_helper_read_vector_x_trans_upper_="/helpers/read_vector_x_incremental_upper.cl";
const std::string FBlasGenerator::k_helper_read_vector_y_trans_upper_="/helpers/read_vector_y_incremental_upper.cl";
const std::string FBlasGenerator::k_helper_read_vector_x_trsv_low_="/helpers/read_write_vector_x_trsv_low.cl";
const std::string FBlasGenerator::k_helper_read_vector_x_trsv_upper_="/helpers/read_write_vector_x_trsv_upper.cl";
const std::string FBlasGenerator::k_helper_write_scalar_="/helpers/write_scalar.cl";
const std::string FBlasGenerator::k_helper_write_integer_="/helpers/write_integer.cl";
const std::string FBlasGenerator::k_helper_write_vector_="/helpers/write_vector.cl";
const std::string FBlasGenerator::k_helper_write_vector_x_y_="/helpers/write_two_vectors.cl";
const std::string FBlasGenerator::k_helper_read_matrix_rowstreamed_tile_row_="/helpers/read_matrix_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_read_matrix_rowstreamed_tile_col_="/helpers/read_matrix_rowstreamed_tile_col.cl";
const std::string FBlasGenerator::k_helper_read_lower_matrix_rowstreamed_tile_row_="/helpers/read_lower_matrix_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_read_upper_matrix_rowstreamed_tile_row_="/helpers/read_upper_matrix_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_read_upper_matrix_rowstreamed_tile_col_="/helpers/read_upper_matrix_rowstreamed_tile_col.cl";
const std::string FBlasGenerator::k_helper_read_reverse_lower_matrix_rowstreamed_tile_col_="/helpers/read_reverse_lower_matrix_rowstreamed_tile_col.cl";
const std::string FBlasGenerator::k_helper_read_reverse_upper_matrix_rowstreamed_tile_row_="/helpers/read_reverse_upper_matrix_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_write_matrix_rowstreamed_tile_row_="/helpers/write_matrix_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_write_lower_matrix_rowstreamed_tile_row_="/helpers/write_lower_matrix_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_write_upper_matrix_rowstreamed_tile_row_="/helpers/write_upper_matrix_rowstreamed_tile_row.cl";


const std::map<std::string,std::string> FBlasGenerator::k_helper_files_= {{k_helper_read_vector_x_, "/helpers/read_vector_x.cl"},{k_helper_read_vector_y_,"/helpers/read_vector_y.cl"},
                                                                        {k_helper_write_scalar_, "/helpers/write_scalar.cl"},{k_helper_write_vector_,"/helpers/write_vector.cl"},
                                                                         {k_helper_read_matrix_rowstreamed_tile_row_,"/helpers/read_matrix_rowstreamed_tile_row.cl"},
                                                                         {k_helper_read_matrix_rowstreamed_tile_col_,"/helpers/read_matrix_rowstreamed_tile_col.cl"},
                                                                         {k_helper_write_matrix_rowstreamed_tile_row_,"/helpers/write_matrix_rowstreamed_tile_row.cl"}};

//level 3 helpers
const std::string FBlasGenerator::k_helper_read_matrix_a_notrans_gemm_="/helpers/read_matrix_a_notrans_gemm.cl";
const std::string FBlasGenerator::k_helper_read_matrix_a_notrans_syrk_="/helpers/read_matrix_a_notrans_syrk.cl";
const std::string FBlasGenerator::k_helper_read_matrix_a_trans_syrk_="/helpers/read_matrix_a_trans_syrk.cl";
const std::string FBlasGenerator::k_helper_read_matrix_a2_trans_syrk_="/helpers/read_matrix_a2_trans_syrk.cl";
const std::string FBlasGenerator::k_helper_read_matrix_a2_notrans_syrk_="/helpers/read_matrix_a2_notrans_syrk.cl";
const std::string FBlasGenerator::k_helper_read_matrix_a_trans_gemm_="/helpers/read_matrix_a_trans_gemm.cl";
const std::string FBlasGenerator::k_helper_read_matrix_b_notrans_gemm_="/helpers/read_matrix_b_notrans_gemm.cl";
const std::string FBlasGenerator::k_helper_read_matrix_b_trans_gemm_="/helpers/read_matrix_b_trans_gemm.cl";
const std::string FBlasGenerator::k_helper_read_matrix_b_trans_syr2k_="/helpers/read_matrix_b_trans_syr2k.cl";
const std::string FBlasGenerator::k_helper_read_matrix_b_notrans_syr2k_="/helpers/read_matrix_b_notrans_syr2k.cl";
const std::string FBlasGenerator::k_helper_read_matrix_b2_notrans_syr2k_="/helpers/read_matrix_b2_notrans_syr2k.cl";
const std::string FBlasGenerator::k_helper_read_matrix_b2_trans_syr2k_="/helpers/read_matrix_b2_trans_syr2k.cl";
const std::string FBlasGenerator::k_helper_write_matrix_gemm_="/helpers/write_matrix_gemm.cl";
const std::string FBlasGenerator::k_helper_write_lower_matrix_syrk_="/helpers/write_lower_matrix_syrk.cl";
const std::string FBlasGenerator::k_helper_write_upper_matrix_syrk_="/helpers/write_upper_matrix_syrk.cl";

const std::string FBlasGenerator::k_helper_read_matrix_as_lower_rowstreamed_tile_row_="/helpers/read_matrix_as_lower_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_read_matrix_as_upper_rowstreamed_tile_row_="/helpers/read_matrix_as_upper_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_read_matrix_as_upper_rowstreamed_tile_col_="/helpers/read_matrix_as_upper_rowstreamed_tile_col.cl";
const std::string FBlasGenerator::k_helper_read_matrix_as_lower_rowstreamed_tile_col_="/helpers/read_matrix_as_lower_rowstreamed_tile_col.cl";

const std::string FBlasGenerator::k_helper_read_matrix_as_symmetric_lower_rowstreamed_tile_row_="/helpers/read_matrix_as_symmetric_lower_rowstreamed_tile_row.cl";
const std::string FBlasGenerator::k_helper_read_matrix_as_symmetric_upper_rowstreamed_tile_col_="/helpers/read_matrix_as_symmetric_upper_rowstreamed_tile_col.cl";
//base names
//channels
const std::string FBlasGenerator::k_channel_in_vector_x_base_name_="channel_in_vector_x_";
const std::string FBlasGenerator::k_channel_in_vector_y_base_name_="channel_in_vector_y_";
const std::string FBlasGenerator::k_channel_in_vector_x_trans_base_name_="channel_in_vector_x_trans_";
const std::string FBlasGenerator::k_channel_in_vector_y_trans_base_name_="channel_in_vector_y_trans_";
const std::string FBlasGenerator::k_channel_in_vector_x_trsv_base_name_="channel_in_vector_x_trsv_";
const std::string FBlasGenerator::k_channel_in_matrix_A_base_name_="channel_in_matrix_A_";
const std::string FBlasGenerator::k_channel_in_matrix_A2_base_name_="channel_in_matrix_A2_";
const std::string FBlasGenerator::k_channel_in_matrix_B_base_name_="channel_in_matrix_B_";
const std::string FBlasGenerator::k_channel_in_matrix_B2_base_name_="channel_in_matrix_B2_";
const std::string FBlasGenerator::k_channel_out_scalar_base_name_="channel_out_scalar_";
const std::string FBlasGenerator::k_channel_out_vector_base_name_="channel_out_vector_";
const std::string FBlasGenerator::k_channel_out_vector_x_base_name_="channel_out_vector_x_";
const std::string FBlasGenerator::k_channel_out_vector_y_base_name_="channel_out_vector_y_";
const std::string FBlasGenerator::k_channel_out_matrix_base_name_="channel_out_matrix_";


//kernels
const std::string FBlasGenerator::k_kernel_read_vector_x_base_name_="kernel_read_vector_x_";
const std::string FBlasGenerator::k_kernel_read_vector_y_base_name_="kernel_read_vector_y_";
const std::string FBlasGenerator::k_kernel_read_vector_x_trans_base_name_="kernel_read_vector_x_trans_";
const std::string FBlasGenerator::k_kernel_read_vector_y_trans_base_name_="kernel_read_vector_y_trans_";
const std::string FBlasGenerator::k_kernel_read_vector_x_trsv_base_name_="kernel_read_write_vector_x_trsv_";
const std::string FBlasGenerator::k_kernel_read_matrix_A_base_name_="kernel_read_matrix_A_";
const std::string FBlasGenerator::k_kernel_read_matrix_A2_base_name_="kernel_read_matrix_A2_";
const std::string FBlasGenerator::k_kernel_read_matrix_B_base_name_="kernel_read_matrix_B_";
const std::string FBlasGenerator::k_kernel_read_matrix_B2_base_name_="kernel_read_matrix_B2_";
const std::string FBlasGenerator::k_kernel_write_scalar_base_name_="kernel_write_scalar_";
const std::string FBlasGenerator::k_kernel_write_vector_base_name_="kernel_write_vector_";
const std::string FBlasGenerator::k_kernel_write_vector_x_y_base_name_="kernel_write_vectors_";
const std::string FBlasGenerator::k_kernel_write_matrix_base_name_="kernel_write_matrix_";
const std::string FBlasGenerator::k_generated_file_header_="//Automatically generated file";



