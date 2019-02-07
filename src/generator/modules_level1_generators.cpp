/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Modules Generator: generators for Level 1 routines
*/

#include <fstream>
#include <string>
#include <set>
#include <map>
#include "../../include/generator/module_generator.hpp"
#include "../../include/commons.hpp"

void ModuleGenerator::Level1Dot(const GeneratorRoutine &r, unsigned int id, std::string output_folder)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_dot.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=ModuleGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons(r,fout);
    addDefineChannelInVectorX(r.getChannelName("x"),fout);
    addDefineChannelInVectorY(r.getChannelName("y"),fout);
    addDefineChannelOutScalar(r.getChannelName("res"),fout);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,r.isKernel(),"KERNEL_NAME",r.getUserName());



    fout.close();

}


void ModuleGenerator::Level1Axpy(const GeneratorRoutine &r, unsigned int id, std::string output_folder)
{

    std::ifstream fin(k_skeleton_folder_ + "/1/streaming_axpy.cl");
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << "(file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");

    bool found_placeholder=ModuleGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons(r,fout);
    addDefineChannelInVectorX(r.getChannelName("x"),fout);
    addDefineChannelInVectorY(r.getChannelName("y"),fout);
    addDefineChannelOutVector(r.getChannelName("res"),fout);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file
    ModuleGenerator::CopyTillEndReplacing(fin,fout,r.isKernel(),"KERNEL_NAME",r.getUserName());



    fout.close();

}

