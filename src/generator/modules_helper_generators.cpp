/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Helpers generation function for Module Generator
*/

#include <fstream>
#include <string>
#include <set>
#include <map>
#include "../../include/generator/module_generator.hpp"
#include "../../include/commons.hpp"

void ModuleGenerator::ReadVectorX(const GeneratorHelper &h, unsigned int id, std::string output_folder)
{

    std::ifstream fin(k_skeleton_folder_ + "/helpers/read_vector_x.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< h.getType() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }
    std::ofstream fout(output_folder+h.getUserName()+".cl");
    fout << k_channel_enable_define_<<std::endl;
    if(h.isDoublePrecision())
        fout << k_double_precision_define_ <<std::endl;
    fout << k_width_define_ << h.getWidth()<<std::endl;
    fout << k_incx_define_<<"" <<h.getStride()<<std::endl;
    fout << k_commons_define_<< std::endl;
    //add the channel definitions (will be included also in the module file)
    fout << k_channel_x_define_<< h.getChannelName()<<std::endl;
    fout << "channel TYPE_T CHANNEL_VECTOR_X __attribute__((depth(W)));"<<std::endl;


    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,true,"READ_VECTOR_X",h.getUserName());

    fout.close();

}


void ModuleGenerator::ReadVectorY(const GeneratorHelper &h, unsigned int id, std::string output_folder)
{

    std::ifstream fin(k_skeleton_folder_ + "/helpers/read_vector_y.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< h.getType() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }
    std::ofstream fout(output_folder+h.getUserName()+".cl");
    fout << k_channel_enable_define_<<std::endl;
    if(h.isDoublePrecision())
        fout << k_double_precision_define_ <<std::endl;
    fout << k_width_define_ << h.getWidth()<<std::endl;
    fout << k_incy_define_<<"" <<h.getStride()<<std::endl;
    fout << k_commons_define_<< std::endl;
    fout << k_channel_y_define_<< h.getChannelName()<<std::endl;
    fout << "channel TYPE_T CHANNEL_VECTOR_Y __attribute__((depth(W)));"<<std::endl;
    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,true,"READ_VECTOR_Y",h.getUserName());

    fout.close();

}


void ModuleGenerator::WriteScalar(const GeneratorHelper &h, unsigned int id, std::string output_folder)
{

    std::ifstream fin(k_skeleton_folder_ + "/helpers/write_scalar.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< h.getType() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }
    std::ofstream fout(output_folder+h.getUserName()+".cl");
    fout << k_channel_enable_define_<<std::endl;
    if(h.isDoublePrecision())
        fout << k_double_precision_define_ <<std::endl;
    fout << k_commons_define_<< std::endl;

    fout << k_channel_scalar_out_define_<< h.getChannelName()<<std::endl;
    fout << "channel TYPE_T CHANNEL_OUT __attribute__((depth(1)));"<<std::endl;


    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,true,"WRITE_SCALAR",h.getUserName());
    fout.close();

}


void ModuleGenerator::WriteVector(const GeneratorHelper &h, unsigned int id, std::string output_folder)
{

    std::ifstream fin(k_skeleton_folder_ + "/helpers/write_vector.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< h.getType() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }
    std::ofstream fout(output_folder+h.getUserName()+".cl");
    fout << k_channel_enable_define_<<std::endl;
    if(h.isDoublePrecision())
        fout << k_double_precision_define_ <<std::endl;
    fout << k_commons_define_<< std::endl;
    fout << k_width_define_ << h.getWidth()<<std::endl;
    fout << k_channel_vector_out_define_<< h.getChannelName()<<std::endl;
    fout << k_incw_define_<<h.getStride()<<std::endl;
    fout << "channel TYPE_T CHANNEL_VECTOR_OUT __attribute__((depth(W)));"<<std::endl;

    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,true,"WRITE_VECTOR",h.getUserName());
    fout.close();

}

void ModuleGenerator::ReadMatrix(const GeneratorHelper &h, unsigned int id, std::string output_folder)
{
    std::string skeleton_name;
    if(h.areTilesRowStreamed() && h.areElementsRowStreamed())
        skeleton_name="/helpers/read_matrix_rowstreamed_tile_row.cl";
    else
        if(!h.areElementsRowStreamed() && h.areElementsRowStreamed())
            skeleton_name="/helpers/read_matrix_rowstreamed_tile_col.cl";
        else
            std::cerr<<RED "Error: not able to generate the code for helper "<<h.getUserName() <<" ("<<h.getType()<<"). Tiles/Elements orders currently not supported." RESET<<std::endl;


    std::ifstream fin(k_skeleton_folder_ +skeleton_name);
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< h.getType() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }
    std::ofstream fout(output_folder+h.getUserName()+".cl");
    fout << k_channel_enable_define_<<std::endl;
    if(h.isDoublePrecision())
        fout << k_double_precision_define_ <<std::endl;
    fout << k_commons_define_<< std::endl;
    fout << k_width_define_ << h.getWidth()<<std::endl;
    fout << k_channel_matrix_A_define_ << h.getChannelName() <<std::endl;
    fout << k_tile_n_size_define_ << h.getTileNsize()<<std::endl;
    fout << k_tile_m_size_define_ << h.getTileMsize()<<std::endl;

    fout << "channel TYPE_T CHANNEL_MATRIX_A __attribute__((depth(W)));"<<std::endl;

    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,true,"READ_MATRIX_A",h.getUserName());
    fout.close();


}


void ModuleGenerator::WriteMatrix(const GeneratorHelper &h, unsigned int id, std::string output_folder)
{
    std::string skeleton_name;
    if(h.areTilesRowStreamed() && h.areElementsRowStreamed())
        skeleton_name="/helpers/write_matrix_rowstreamed_tile_row.cl";
    else
        if(!h.areTilesRowStreamed() && !h.areElementsRowStreamed())
            skeleton_name="/helpers/write_matrix_colstreamed_tile_col.cl";
        else
            std::cerr<<RED "Error: not able to generate the code for helper "<<h.getUserName() <<" ("<<h.getType()<<"). Tiles/Elements orders currently not supported." RESET<<std::endl;

    std::ifstream fin(k_skeleton_folder_ +skeleton_name);
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< h.getType() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }
    std::ofstream fout(output_folder+h.getUserName()+".cl");
    fout << k_channel_enable_define_<<std::endl;
    if(h.isDoublePrecision())
        fout << k_double_precision_define_ <<std::endl;
    fout << k_commons_define_<< std::endl;
    fout << k_width_define_ << h.getWidth()<<std::endl;
    fout << k_channel_matrix_out_define_ << h.getChannelName() <<std::endl;
    fout << k_tile_n_size_define_ << h.getTileNsize()<<std::endl;
    fout << k_tile_m_size_define_ << h.getTileMsize()<<std::endl;

    fout << "channel TYPE_T CHANNEL_MATRIX_OUT __attribute__((depth(W)));"<<std::endl;

    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,true,"WRITE_MATRIX",h.getUserName());
    fout.close();


}
