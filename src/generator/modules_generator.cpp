/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Modules Generator

*/
#include <rapidjson/document.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <exception>
#include "../../include/generator/json_parser.hpp"
#include "../../include/generator/module_generator.hpp"

/*This is the main program for code generation*/
int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cerr<< "Usage: " <<argv[0] << " <json file> [output directory]"<<std::endl;
        return -1;
    }

    const std::string program_path=argv[1];
    std::string output_dir;
    if(argc==3)
        output_dir=argv[2];
    else
        output_dir="/tmp/";

    if(output_dir[output_dir.length()-1]!='/')
        output_dir.append("/");
    std::cout << GREEN "FBLAS Modules Code generator: parsing file "<<program_path << "...." RESET <<std::endl;
    JSONParser parser(program_path,1);

    const std::vector<GeneratorRoutine> &rout=parser.getRoutines();
    const std::vector<GeneratorHelper> &helpers=parser.getHelpers();
    std::cout << GREEN "FBLAS Code generator: code for "<<rout.size()<<" routine(s) will be generated (output dir "<<output_dir<<")" RESET <<std::endl;

    ModuleGenerator gen;
    gen.GenerateCode(rout,helpers,output_dir);
    std::cout << GREEN "FBLAS Code generator: generation completed." RESET <<std::endl;
}
