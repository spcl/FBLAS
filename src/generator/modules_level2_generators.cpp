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

void ModuleGenerator::Level2Gemv(const GeneratorRoutine &r, unsigned int id, std::string output_folder)
{

    //check the version
    std::string skeleton_name;
    if((r.areTilesARowStreamed() && r.areElementsARowStreamed() && r.getTransposeA()==FBLAS_NO_TRANSPOSED) || (!r.areElementsARowStreamed() && !r.areTilesARowStreamed() && r.getTransposeA()==FBLAS_TRANSPOSED))
    {
        skeleton_name="2/streaming_gemv_v1.cl";
    }
    else
    {
        std::cerr<<RED "Error: not able to generate the code for "<<r.getUserName() <<" ("<<r.getBlasName()<<"). Tiles/Elements orders/Transposition currently not supported." RESET<<std::endl;
        return;
    }

    std::ifstream fin(k_skeleton_folder_ + skeleton_name);
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }

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
    addDefineChannelInMatrixA(r.getChannelName("A"),fout);
    addTileN(r,fout);
    addTileM(r,fout);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file and replace header
    ModuleGenerator::CopyTillEndReplacing(fin,fout,r.isKernel(),"KERNEL_NAME",r.getUserName());



    fout.close();

}


void ModuleGenerator::Level2Ger(const GeneratorRoutine &r, unsigned int id, std::string output_folder)
{

    //check the version
    std::string skeleton_name;
    if(r.areTilesARowStreamed() && r.areElementsARowStreamed())
    {
        skeleton_name="2/streaming_ger_v1.cl";
    }
    else
    {
        std::cerr<<RED "Error: not able to generate the code for "<<r.getUserName() <<" ("<<r.getBlasName()<<"). Tiles/Elements orders/Transposition currently not supported." RESET<<std::endl;
        return;
    }

    std::ifstream fin(k_skeleton_folder_ + skeleton_name);
    if(!fin.is_open()){
        std::cerr << "Error in opening skeleton file for "<< r.getBlasName() << " (file path: "<<k_skeleton_folder_<<")"<<std::endl;
        return;
    }

    std::ofstream fout(output_folder+r.getUserName()+".cl");
    if(!fout.is_open()){
        std::cerr << "Error in opening output file for "<< r.getUserName() << " (file path: "<<output_folder<<")"<<std::endl;
        return;
    }

    bool found_placeholder=ModuleGenerator::CopyHeader(fin,fout);
    if(!found_placeholder)
    {
        std::cerr<<"Placeholder not found in skeleton definition. Generation for this routine failed!" << std::endl;
        return;
    }

    addCommons(r,fout);
    addDefineChannelInVectorX(r.getChannelName("x"),fout);
    addDefineChannelInVectorY(r.getChannelName("y"),fout);
    addDefineChannelOutMatrix(r.getChannelName("res"),fout);
    addDefineChannelInMatrixA(r.getChannelName("A"),fout);
    addTileN(r,fout);
    addTileM(r,fout);

    //platform
    if(r.getArchitecture()==FBLAS_STRATIX_10)
        fout << k_stratix_10_platform_define_ <<std::endl;
    else
        fout << k_arria_10_platform_define_<<std::endl;

    //copy the rest of the file and replace header
    ModuleGenerator::CopyTillEndReplacing(fin,fout,r.isKernel(),"KERNEL_NAME",r.getUserName());



    fout.close();

}

