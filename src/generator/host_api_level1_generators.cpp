/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Code Generator: generators for Level 1 routines
*/
#include <fstream>
#include <string>
#include <set>
#include <map>
#include "../../include/generator/fblas_generator.hpp"
#include "../../include/commons.hpp"


void FBlasGenerator::Level1Swap(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_swap.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons("swap",r,json_writer,fout);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelOutVectorXY(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperWriteVectorXY(id,json_writer,fout);

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
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_vector_x_y_,fout);

    fout.close();
}



void FBlasGenerator::Level1Rot(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_rot.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons("rot",r,json_writer,fout);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelOutVectorXY(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperWriteVectorXY(id,json_writer,fout);

    closeRoutineItem(json_writer);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_y_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_vector_x_y_,fout);

    fout.close();

}


void FBlasGenerator::Level1Rotm(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_rotm.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons("rotm",r,json_writer,fout);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelOutVectorXY(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperWriteVectorXY(id,json_writer,fout);

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
    FBlasGenerator::CopyHelper(k_skeleton_folder_ + k_helper_write_vector_x_y_,fout);

    fout.close();

}


void FBlasGenerator::Level1Rotg(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    //For the moment being rotg is realized directly into the API (it's not worth calling an opencl kernel)
    //Therefore the output of the generation will be just an empty file
    //This has been done for compatibility with respect to the rest of the library and to leave space for further modifications
    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;
    fout << "__kernel void empty_"<<id<<"(){ /*empty kernel, used for rotg*/}"<<std::endl;

    json_writer.StartObject();
    json_writer.Key(k_json_field_blas_name);
    json_writer.String("rotg");
    json_writer.Key(k_json_field_user_name);
    json_writer.String(r.getUserName().c_str());
    //precision
    json_writer.Key(k_json_field_precision);
    if(r.isDoublePrecision())
        json_writer.String("double");
    else
        json_writer.String("single");

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    closeRoutineItem(json_writer);

    fout.close();
}

void FBlasGenerator::Level1Rotmg(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    //For the moment being rotmg is realized directly into the API (it's not worth calling an opencl kernel)
    //Therefore the output of the generation will be just an empty file
    //This has been done for compatibility with respect to the rest of the library and to leave space for further modifications
    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;
    fout << "__kernel void empty_"<<id<<"(){ /*empty kernel, used for rotmg*/}"<<std::endl;
    json_writer.StartObject();
    json_writer.Key(k_json_field_blas_name);
    json_writer.String("rotmg");
    json_writer.Key(k_json_field_user_name);
    json_writer.String(r.getUserName().c_str());
    //precision
    json_writer.Key(k_json_field_precision);
    if(r.isDoublePrecision())
        json_writer.String("double");
    else
        json_writer.String("single");

    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    closeRoutineItem(json_writer);

    fout.close();
}


void FBlasGenerator::Level1Axpy(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_axpy.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons("axpy",r,json_writer,fout);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);
    addDefineIncW(r,r.getIncy(),json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelOutVector(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
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
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_y_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_vector_,fout);

    fout.close();

}


void FBlasGenerator::Level1Dot(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_dot.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons("dot",r,json_writer,fout);

    //incx and y
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelInVectorY(id,fout);
    addDefineChannelOutScalar(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperReadVectorY(id,json_writer,fout);
    addDefineAndItemHelperWriteScalar(id,json_writer,fout);

    closeRoutineItem(json_writer);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_files_.at(k_helper_read_vector_y_),fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_scalar_,fout);

    fout.close();

}


void FBlasGenerator::Level1Scal(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_scal.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //fill at the same time the json
    addCommons("scal",r,json_writer,fout);

    //tailored code generation
    addIncX(r,json_writer,fout);
    addDefineIncW(r,r.getIncx(),json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelOutVector(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
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
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_files_.at(k_helper_write_vector_),fout);

    fout.close();

}


void FBlasGenerator::Level1Asum(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_asum.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //fill at the same time the json
    addCommons("asum",r,json_writer,fout);

    //tailored code generation
    addIncX(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelOutScalar(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperWriteScalar(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_scalar_,fout);

    fout.close();

}

void FBlasGenerator::Level1Iamax(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_iamax.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //fill at the same time the json
    addCommons("iamax",r,json_writer,fout);

    //tailored code generation
    addIncX(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelOutScalar(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperWriteScalar(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_integer_,fout);

    fout.close();

}

void FBlasGenerator::Level1Nrm2(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_nrm2.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;

    bool found_placeholder=FBlasGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    //fill at the same time the json
    addCommons("nrm2",r,json_writer,fout);

    //tailored code generation
    addIncX(r,json_writer,fout);

    //namings
    addDefineKernelName(r,fout);
    addDefineChannelInVectorX(id,fout);
    addDefineChannelOutScalar(id,fout);

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperWriteScalar(id,json_writer,fout);

    closeRoutineItem(json_writer);
    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    FBlasGenerator::CopyTillEnd(fin,fout);

    //copy the helpers
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_scalar_,fout);

    fout.close();

}



void FBlasGenerator::Level1Copy(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
{
    //Copy is created by using the vector reader and writer helpers
    //The generation is very peculiar
    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }
    fout << k_generated_file_header_<<std::endl;
    fout << "#pragma OPENCL EXTENSION cl_intel_channels : enable" <<std::endl;

    addCommons("copy",r,json_writer,fout);

    //incx and y (actually y is incw, the stride used for the writing)
    addIncX(r,json_writer,fout);
    addIncY(r,json_writer,fout);
    addDefineIncW(r,r.getIncy(),json_writer,fout);

    //namings
    //there is no kernel name here and the reading and writing channel must be the same
    std::string name=k_channel_in_vector_x_base_name_+std::to_string(id);
    fout << k_channel_x_define_ << name<<std::endl;
    fout << k_channel_vector_out_define_ << name <<std::endl;

    //Helper kernels names
    addDefineAndItemHelperReadVectorX(id,json_writer,fout);
    addDefineAndItemHelperWriteVector(id,json_writer,fout);
    closeRoutineItem(json_writer);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;
    //add the include
    fout <<k_commons_define_ << std::endl;
    //add the channel defiitions

    fout << "channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));" <<std::endl;


    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_read_vector_x_,fout);
    FBlasGenerator::CopyTillEnd(k_skeleton_folder_ + k_helper_write_vector_,fout);

    fout.close();

}

