/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Code Generator: generators for Level 2 routines
*/
#include <fstream>
#include <string>
#include <set>
#include <map>
#include "../../include/generator/fblas_generator.hpp"
#include "../../include/commons.hpp"



void FBlasGenerator::Level2Gemv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    //We have multiple cases
    FblasOrder ord=r.getOrder();
    FblasTranspose trans=r.getTransposeA();

    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_NO_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_gemv_v1.cl");
    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_gemv_v2.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("gemv",r,json_writer,fout);


    //tailored code generation
    //For GEMV we need a reader for x (with repetitions), one for y, one for the matrix the actual routine and the sink
    //First of all let's fill the MACROS properly (and the json at the same time)

    addTileN(r,json_writer,fout);
    addTileM(r,json_writer,fout);
    addOrder(r,json_writer);
    addTranspose(r,json_writer);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);
    addDefineIncW(r,r.getIncy(),json_writer,fout);

    //json_writer.Key(k_json_field_lda);
    //json_writer.Int(r.getLda());

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutVector(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperWriteVector(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_y_,fout);

    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_NO_TRANSPOSED)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_rowstreamed_tile_row_,fout);
    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_TRANSPOSED)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_rowstreamed_tile_col_,fout);

    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_vector_,fout);

    fout.close();
}

void FBlasGenerator::Level2Trmv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    //For the moment being this routine is realized using the classical gemv
    //matrix A is read properly (avoiding the part that should be not referenced)
    //since we are using tiling, x (that is input and output in this case) should be copied
    //Note: we need a reader for y (even if we will use beta=0)

    //We have multiple cases
    FblasOrder ord=r.getOrder();
    FblasUpLo uplo=r.getUplo();

    FblasTranspose trans=r.getTransposeA();

    std::ifstream fin;
    //We have to distinguish between two cases
    //this is emulated using standard gemv with a clever reading
    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_NO_TRANSPOSED )
        fin.open(k_skeleton_folder_ + "/2/streaming_gemv_v1.cl");
    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_gemv_v2.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("trmv",r,json_writer,fout);

    addTileN(r,json_writer,fout);
    //add tile M with the same vale of tile N
    json_writer.Key(k_json_field_tile_m_size);
    fout << k_tile_m_size_define_;
    unsigned int size= (r.getTileNsize()!=0)? r.getTileNsize() : k_default_tiling_;
    if(r.getWidth() != 0 && size%r.getWidth() != 0)
        size = r.getWidth() * 4 ;
    fout << size <<std::endl;
    json_writer.Uint(size);

    addOrder(r,json_writer);
    addTranspose(r,json_writer);
    addUpLo(r,json_writer);
    //incx: must be the same for input (incx) and output (incw)
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout); //added because otherwise it doesn't compile
    addDefineIncW(r,r.getIncx(),json_writer,fout);


    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutVector(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperWriteVector(id,json_writer,fout);

    closeRoutineItem(json_writer);
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;
    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_y_,fout);


    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_NO_TRANSPOSED && uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_as_lower_rowstreamed_tile_row_,fout);

    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_NO_TRANSPOSED && uplo==FBLAS_UPPER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_as_upper_rowstreamed_tile_row_,fout);


    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_TRANSPOSED && uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_as_lower_rowstreamed_tile_col_,fout);

    if(ord==FBLAS_ROW_MAJOR && trans==FBLAS_TRANSPOSED && uplo==FBLAS_UPPER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_as_upper_rowstreamed_tile_col_,fout);

    //todo the other version

    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_vector_,fout);

    fout.close();
}


void FBlasGenerator::Level2Symv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    //For the moment being The routine is realized by using the classical GEMV
    FblasOrder ord=r.getOrder();
    FblasUpLo uplo=r.getUplo();

    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR)
        fin.open(k_skeleton_folder_ + "/2/streaming_gemv_v1.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("symv",r,json_writer,fout);


    //tailored code generation
    //For GEMV we need a reader for x (with repetitions), one for y, one for the matrix the actual routine and the sink
    //First of all let's fill the MACROS properly (and the json at the same time)

    addTileN(r,json_writer,fout);
    //add tile M with the same vale of tile N
    json_writer.Key(k_json_field_tile_m_size);
    fout << k_tile_m_size_define_;
    unsigned int size= (r.getTileNsize()!=0)? r.getTileNsize() : k_default_tiling_;
    if(r.getWidth() != 0 && size%r.getWidth() != 0)
        size = r.getWidth() * 4 ;
    fout << size <<std::endl;
    json_writer.Uint(size);

    addOrder(r,json_writer);
    addTranspose(r,json_writer);
    addUpLo(r,json_writer);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);
    addDefineIncW(r,r.getIncy(),json_writer,fout);

    //json_writer.Key(k_json_field_lda);
    //json_writer.Int(r.getLda());

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutVector(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperWriteVector(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_y_,fout);

    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER )
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_as_symmetric_lower_rowstreamed_tile_row_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER )
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_as_symmetric_upper_rowstreamed_tile_col_,fout);
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_vector_,fout);

    fout.close();
}

void FBlasGenerator::Level2Ger(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    FblasOrder ord=r.getOrder();

    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR)
        fin.open(k_skeleton_folder_ + "/2/streaming_ger_v1.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("ger",r,json_writer,fout);


    //tailored code generation
    //For GER we need a reader for x and y, the matrix and a writer for the matrix
    //In case of RowMajor, y must be resent multiple times

    addTileN(r,json_writer,fout);
    addTileM(r,json_writer,fout);
    addOrder(r,json_writer);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);

    //json_writer.Key(k_json_field_lda);
    //json_writer.Int(r.getLda());

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutMatrix(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperWriteMatrix(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_files_.at(k_helper_read_vector_x_),fout);
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_files_.at(k_helper_read_vector_y_),fout);

    if(ord==FBLAS_ROW_MAJOR )
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_files_.at(k_helper_read_matrix_rowstreamed_tile_row_),fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_files_.at(k_helper_write_matrix_rowstreamed_tile_row_),fout);
    }

    fout.close();

}

void FBlasGenerator::Level2Syr(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    FblasOrder ord=r.getOrder();
    FblasUpLo uplo=r.getUplo();
    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
        fin.open(k_skeleton_folder_ + "/2/streaming_syr_v1.cl");
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
        fin.open(k_skeleton_folder_ + "/2/streaming_syr_v2.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("syr",r,json_writer,fout);


    //tailored code generation
    //For GER we need a reader for x and y, the matrix and a writer for the matrix
    //In case of RowMajor, y must be resent multiple times

    addTileN(r,json_writer,fout);
    addOrder(r,json_writer);
    addUpLo(r,json_writer);

    //incx and y
    addIncX(r,json_writer,fout);

    //json_writer.Key(k_json_field_lda);
    //json_writer.Int(r.getLda());

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorXTrans(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutMatrix(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorXTrans(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperWriteMatrix(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trans_low_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trans_upper_,fout);

    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_lower_matrix_rowstreamed_tile_row_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_lower_matrix_rowstreamed_tile_row_,fout);
    }

    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_upper_matrix_rowstreamed_tile_row_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_upper_matrix_rowstreamed_tile_row_,fout);
    }
    fout.close();

}


void FBlasGenerator::Level2Syr2(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    FblasOrder ord=r.getOrder();
    FblasUpLo uplo=r.getUplo();
    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
        fin.open(k_skeleton_folder_ + "/2/streaming_syr2_v1.cl");
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
        fin.open(k_skeleton_folder_ + "/2/streaming_syr2_v2.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("syr2",r,json_writer,fout);


    //tailored code generation
    //For GER we need a reader for x and y, the matrix and a writer for the matrix
    //In case of RowMajor, y must be resent multiple times

    addTileN(r,json_writer,fout);
    addOrder(r,json_writer);
    addUpLo(r,json_writer);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);

    //json_writer.Key(k_json_field_lda);
    //json_writer.Int(r.getLda());

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorXTrans(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelInVectorYTrans(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutMatrix(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorXTrans(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperReadVectorYTrans(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperWriteMatrix(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trans_low_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trans_upper_,fout);

    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_y_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_y_trans_low_,fout);
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_y_trans_upper_,fout);

    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_lower_matrix_rowstreamed_tile_row_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_lower_matrix_rowstreamed_tile_row_,fout);
    }

    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_upper_matrix_rowstreamed_tile_row_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_upper_matrix_rowstreamed_tile_row_,fout);
    }
    fout.close();

}

void FBlasGenerator::Level2Trsv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{


    FblasOrder ord=r.getOrder();
    FblasUpLo uplo=r.getUplo();
    FblasTranspose trans=r.getTransposeA();
    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER && trans==FBLAS_NO_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_trsv_v1.cl");
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER && trans==FBLAS_NO_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_trsv_v2.cl");
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER && trans==FBLAS_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_trsv_v3.cl");
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER && trans==FBLAS_TRANSPOSED)
        fin.open(k_skeleton_folder_ + "/2/streaming_trsv_v4.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("trsv",r,json_writer,fout);


    //tailored code generation
    //For GER we need a reader for x and y, the matrix and a writer for the matrix
    //In case of RowMajor, y must be resent multiple times

    addTileN(r,json_writer,fout);
    addOrder(r,json_writer);
    addTranspose(r,json_writer);

    //incx and y
    addIncX(r,json_writer,fout);

    //json_writer.Key(k_json_field_lda);
    //json_writer.Int(r.getLda());

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorXTrsv(id,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelOutVector(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorXTrsv(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    //here we don't need the vector writer kernel: it will be the reader
    //to write the result back in memory

    closeRoutineItem(json_writer);
    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER && trans==FBLAS_NO_TRANSPOSED)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trsv_low_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_lower_matrix_rowstreamed_tile_row_,fout);
    }
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER && trans==FBLAS_NO_TRANSPOSED)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trsv_upper_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_reverse_upper_matrix_rowstreamed_tile_row_,fout);
    }
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_UPPER && trans==FBLAS_TRANSPOSED)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trsv_low_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_upper_matrix_rowstreamed_tile_col_,fout);
    }
    if(ord==FBLAS_ROW_MAJOR && uplo==FBLAS_LOWER && trans==FBLAS_TRANSPOSED)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_vector_x_trsv_upper_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_reverse_lower_matrix_rowstreamed_tile_col_,fout);
    }


    fout.close();

}
