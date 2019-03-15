/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Code Generator: generators for Level 3 routines

    Please Note: up to now, Level 3 routines are not streamed, so there is no really need of including
   tiling information into the generated JSON. By the way we will keep them for further developments


*/
#include <fstream>
#include <string>
#include <set>
#include <map>
#include "../../include/generator/fblas_generator.hpp"
#include "../../include/commons.hpp"

void FBlasGenerator::Level3Gemm(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    FblasOrder ord=r.getOrder();
    FblasTranspose transa=r.getTransposeA();
    FblasTranspose transb=r.getTransposeB();
    std::ifstream fin;
    if(ord==FBLAS_ROW_MAJOR )
        fin.open(k_skeleton_folder_ + "/3/gemm.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("gemm",r,json_writer,fout);

    //add tile size and info about transpose
    addTile(r,json_writer,fout);
    addTranspose(r,json_writer);
    addTransposeB(r,json_writer);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelInMatrixB(id,fout);
    addDefineChannelOutMatrix(id,fout);


    //Helper kernels names
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixB(id,json_writer,fout);
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
    if(transa==FBLAS_NO_TRANSPOSED)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a_notrans_gemm_,fout);
    else
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a_trans_gemm_,fout);

    if(transb==FBLAS_NO_TRANSPOSED)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_b_notrans_gemm_,fout);
    else
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_b_trans_gemm_,fout);

    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_matrix_gemm_,fout);
    fout.close();
}



void FBlasGenerator::Level3Syrk(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    FblasOrder ord=r.getOrder();
    FblasTranspose trans=r.getTransposeA();
    FblasUpLo uplo=r.getUplo();
    std::ifstream fin;
    if(ord==FBLAS_ROW_MAJOR )
        fin.open(k_skeleton_folder_ + "/3/syrk.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //In this case we have the computational kernel, read A, read A2 and writer

    //add common fields and defines
    addCommons("syrk",r,json_writer,fout);

    //add tile size and info about transpose
    addTile(r,json_writer,fout);
    addTranspose(r,json_writer);
    addUpLo(r,json_writer);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelInMatrixA2(id,fout);
    addDefineChannelOutMatrix(id,fout);


    //Helper kernels names
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA2(id,json_writer,fout);
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
    if(trans==FBLAS_NO_TRANSPOSED)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a_notrans_syrk_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a2_trans_syrk_,fout);
    }
    else
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a_trans_syrk_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a2_notrans_syrk_,fout);
    }

    if(uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_lower_matrix_syrk_,fout);
    else
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_upper_matrix_syrk_,fout);

    fout.close();
}



void FBlasGenerator::Level3Syr2k(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    FblasOrder ord=r.getOrder();
    FblasTranspose trans=r.getTransposeA();
    FblasUpLo uplo=r.getUplo();
    std::ifstream fin;
    if(ord==FBLAS_ROW_MAJOR )
        fin.open(k_skeleton_folder_ + "/3/syr2k.cl");

    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //In this case we have the computational kernel, read A, read B, read A2 and read B2 and writer
    //the kernels that read the input matrices will have a parameter to change how to read depending
    //on if C is lower or upper
    //we can reuse some of the helpers of syrk

    //add common fields and defines
    addCommons("syr2k",r,json_writer,fout);

    //add tile size and info about transpose
    addTile(r,json_writer,fout);
    addTranspose(r,json_writer);
    addUpLo(r,json_writer);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInMatrixA(id,fout);
    addDefineChannelInMatrixA2(id,fout);
    addDefineChannelInMatrixB(id,fout);
    addDefineChannelInMatrixB2(id,fout);
    addDefineChannelOutMatrix(id,fout);


    //Helper kernels names
    addDefineAndItemHelperReadMatrixA(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixA2(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixB(id,json_writer,fout);
    addDefineAndItemHelperReadMatrixB2(id,json_writer,fout);

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
    if(trans==FBLAS_NO_TRANSPOSED)
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a_notrans_syrk_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a2_trans_syrk_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_b_trans_syr2k_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_b2_notrans_syr2k_,fout);
    }
    else
    {
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a_trans_syrk_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_a2_notrans_syrk_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_b_notrans_syr2k_,fout);
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_read_matrix_b2_trans_syr2k_,fout);
    }

    if(uplo==FBLAS_LOWER)
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_lower_matrix_syrk_,fout);
    else
        FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_upper_matrix_syrk_,fout);

    fout.close();
}


void FBlasGenerator::Level3Trsm(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    //We have multiple cases
    FblasOrder ord=r.getOrder();
    FblasTranspose trans=r.getTransposeA();
    FblasSide side=r.getSide();
    FblasUpLo uplo=r.getUplo();

    std::ifstream fin;
    //We have to distinguish between two cases
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_LEFT && trans==FBLAS_NO_TRANSPOSED && uplo==FBLAS_LOWER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v1.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_LEFT && trans==FBLAS_NO_TRANSPOSED && uplo==FBLAS_UPPER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v2.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_LEFT && trans==FBLAS_TRANSPOSED && uplo==FBLAS_LOWER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v3.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_LEFT && trans==FBLAS_TRANSPOSED && uplo==FBLAS_UPPER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v4.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_RIGHT && trans==FBLAS_NO_TRANSPOSED && uplo==FBLAS_LOWER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v5.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_RIGHT && trans==FBLAS_NO_TRANSPOSED && uplo==FBLAS_UPPER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v6.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_RIGHT && trans==FBLAS_TRANSPOSED && uplo==FBLAS_LOWER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v7.cl");
    if(ord==FBLAS_ROW_MAJOR && side == FBLAS_RIGHT && trans==FBLAS_TRANSPOSED && uplo==FBLAS_UPPER )
        fin.open(k_skeleton_folder_ + "/3/trsm_v8.cl");



    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //add common fields and defines
    addCommons("trsm",r,json_writer,fout);


    //tailored code generation
    addTileM(r,json_writer,fout);
    addOrder(r,json_writer);
    addTranspose(r,json_writer);
    addUpLo(r,json_writer);
    addSide(r,json_writer);

    //namings
    addDefineKernelName(r,fout);

    closeRoutineItem(json_writer);
    //platform

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    fout.close();
}
