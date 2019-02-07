/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Modules generator: modules code generator
*/

#ifndef MODULES_GENERATOR_HPP
#define MODULES_GENERATOR_HPP
#include <vector>
#include <map>
#include <functional>
#include <fstream>
#include <regex>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include "generator_routine.hpp"
#include "generator_helper.hpp"
#include "generator_commons.hpp"

/**
 * @brief The FBlasGenerator class implements the FBlas code generator.
 *  Given a set of (parsed) routines it will generate the source code suitable for compilation
 */
class ModuleGenerator {
public:
    ModuleGenerator()
    {
        fillGenerationFunctions();
    }


    void GenerateCode(const std::vector<GeneratorRoutine> &routines, const std::vector<GeneratorHelper> &helpers,std::string output_dir)
    {
        if(routines.size()==0)
        {
            std::cout << RED "No routines to generate" RESET <<std::endl;
            return;
        }

        //dobbiamo avere un insieme di channel descriptors
        for(int i=0; i< routines.size();i++)
        {
            std::cout << "Generating: "<<routines[i].getUserName()<<std::endl;
            //For each routine the generated code will looks like:
            //1) Defines (names, precision, channels, width, ...)
            //2) Main computational kernel
            //3) Helper kernels (read/write from /to memory)
            if(generation_functions_.find(routines[i].getBlasName())!=generation_functions_.end())
                generation_functions_[routines[i].getBlasName()](routines[i],i,output_dir);
            else
                std::cout << "Generation function not present " << routines[i].getBlasName()<<std::endl;
        }

        for(int i=0;i<helpers.size();i++)
        {
            if(generation_helpers_.find(helpers[i].getType())!=generation_helpers_.end())
                generation_helpers_[helpers[i].getType()](helpers[i],i,output_dir);
            else
                std::cout << "Helper generation function not present " << helpers[i].getType()<<std::endl;
        }


    }


private:

    //Level 1
    static void Level1Dot(const GeneratorRoutine &r, unsigned int id, std::string output_folder);
    static void Level1Axpy(const GeneratorRoutine &r, unsigned int id, std::string output_folder);

    //Level2
    static void Level2Gemv(const GeneratorRoutine &r, unsigned int id, std::string output_folder);
    static void Level2Ger(const GeneratorRoutine &r, unsigned int id, std::string output_folder);


    //Helpers
    static void ReadVectorX(const GeneratorHelper &h, unsigned int id, std::string output_folder);
    static void ReadVectorY(const GeneratorHelper &h, unsigned int id, std::string output_folder);
    static void WriteScalar(const GeneratorHelper &h, unsigned int id, std::string output_folder);
    static void WriteVector(const GeneratorHelper &h, unsigned int id, std::string output_folder);
    static void ReadMatrix(const GeneratorHelper &h, unsigned int id, std::string output_folder);
    static void WriteMatrix(const GeneratorHelper &h, unsigned int id, std::string output_folder);



    //Generation functions
    void fillGenerationFunctions()
    {
        //	    parsing_functions_.insert(std::make_pair("sdot",BlasParser::Level1Sdot));
        generation_functions_["axpy"]=ModuleGenerator::Level1Axpy;
        generation_functions_["dot"]=ModuleGenerator::Level1Dot;

        //level 2
        generation_functions_["gemv"]=ModuleGenerator::Level2Gemv;
        generation_functions_["ger"]=ModuleGenerator::Level2Ger;

        //helpers
        generation_helpers_["read vector x"]=ModuleGenerator::ReadVectorX;
        generation_helpers_["read vector y"]=ModuleGenerator::ReadVectorY;
        generation_helpers_["write vector"]=ModuleGenerator::WriteVector;
        generation_helpers_["write scalar"]=ModuleGenerator::WriteScalar;
        generation_helpers_["read matrix"]=ModuleGenerator::ReadMatrix;
        generation_helpers_["write matrix"]=ModuleGenerator::WriteMatrix;
    }

    /**
     * @brief addCommons starts to fill up the generating file and the routine genereated json item
     *      with characteristics that are common to all the routines (name, user_name, precision, width)
     * @param blas_name
     * @param r
     * @param fout
     */
    static void addCommons(const GeneratorRoutine &r,std::ofstream &fout)
    {
        if(r.isDoublePrecision())
            fout << k_double_precision_define_ <<std::endl;
        //width: how to setup this information depends on the type of routine
        if(!r.has2DComputationalTile())
        {
            fout << k_width_define_;
            unsigned int width=(r.getWidth()!=0)?r.getWidth():k_default_width_;
            fout << width <<std::endl;
        }
        else
        {
            unsigned int width_x=(r.getWidthX()!=0)?r.getWidthX():k_default_2d_width_;
            unsigned int width_y=(r.getWidthY()!=0)?r.getWidthY():k_default_2d_width_;
            //Remember width_x are the computational tile cols, while width_y is the number of rows
            fout << k_ctile_rows_define_;
            fout << width_y <<std::endl;
            fout << k_ctile_cols_define_;
            fout << width_x <<std::endl;
        }
    }


    /**
     * @brief addTileN add the informations about the Tile Size referring to N
     *
     */
    static void addTileN(const GeneratorRoutine &r, std::ofstream &fout)
    {
        fout << k_tile_n_size_define_;
        //use default tile size if tile size was not defined by the user
        //or a multiple of the width if the default size is not a multiple
        unsigned int size= (r.getTileNsize()!=0)? r.getTileNsize() : k_default_tiling_;
        if(r.getWidth() != 0 && (size%r.getWidth() != 0 || size<r.getWidth()))
        {
            size = r.getWidth() * 4 ;
            std::cout << "Tile N size for routine " << r.getUserName() << " was not a multiple of the widthor was smaller than width. Set at: "<<size<<std::endl;
        }
        fout << size <<std::endl;
    }

    /**
     * @brief addTileM add the informations about the Tile Size referring to M
     *  to both the generating kernel and routine json definition
     */
    static void addTileM(const GeneratorRoutine &r,  std::ofstream &fout)
    {
        fout << k_tile_m_size_define_;
        unsigned int size= (r.getTileMsize()!=0)? r.getTileMsize() : k_default_tiling_;
        if(r.getWidth() != 0 && (size%r.getWidth() != 0 || size<r.getWidth()))
        {
            size = r.getWidth() * 4 ;
            std::cout << "Tile M size for routine " << r.getUserName() << " was not a multiple of the width. Set at: "<<size<<std::endl;
        }
        fout << size <<std::endl;
    }
    /**
     * @brief addDefineAndItem add a given define and a given item, both of them with a given string as value
     */
    static void addDefineAndItem(std::string define_name, std::string value,  std::ofstream &fout)
    {
        fout << define_name<<value<<std::endl;
    }

    /**
     * @brief addDefineChannelInVectorX add the define for input channel vector X
     */
    static void addDefineChannelInVectorX(std::string channel_name, std::ofstream &fout)
    {
        fout << k_channel_x_define_ << channel_name<<std::endl;
    }

    /**
     * @brief addDefineChannelInVectorY add the define for input channel vector Y to the generating opencl kernel
     */
    static void addDefineChannelInVectorY(std::string channel_name, std::ofstream &fout)
    {
        fout << k_channel_y_define_ << channel_name<<std::endl;
    }

    /**
     * @brief addDefineChannelOutVector add define for channel out vector
     */
    static void addDefineChannelOutVector(std::string channel_name, std::ofstream &fout)
    {
        fout << k_channel_vector_out_define_ << channel_name<<std::endl;
    }
    /**
     * @brief addDefineChannelOutScalar add define for channel out scalar
     */
    static void addDefineChannelOutScalar(std::string channel_name,  std::ofstream &fout)
    {
        fout << k_channel_scalar_out_define_ << channel_name<<std::endl;
    }

    static void addDefineChannelOutMatrix(std::string channel_name,  std::ofstream &fout)
    {
        fout << k_channel_matrix_out_define_ << channel_name<<std::endl;
    }

    static void addDefineChannelInMatrixA(std::string channel_name, std::ofstream &fout)
    {
        fout << k_channel_matrix_A_define_ << channel_name<<std::endl;
    }


    /**
     * @brief CopyHeader copies the header of the file till the placeholder is found. For modules
     *      we maintain the first block of comments since it contains useful information
     * @param fin
     * @param fout
     * @return true if the placeholder have been found, false otherwise
     */
    static bool CopyHeader(std::ifstream &fin, std::ofstream &fout)
    {
        const std::string kPlaceholder_start="//FBLAS_PARAMETERS_START";
        const std::string kPlaceholder_end="//FBLAS_PARAMETERS_END";
        bool first_comment_block_skipped=true;
        std::string line;
        fout << k_generated_file_header_<<std::endl;

        while(std::getline(fin,line))
        {
            if(!first_comment_block_skipped)
            {
                if(line.find("/*")!=std::string::npos)
                {
                    //skip till the comment is finished
                    while(std::getline(fin,line) && line.find("*/")== std::string::npos);
                    first_comment_block_skipped=true;
                    continue;
                }
            }

            if(line.find(kPlaceholder_start)!= std::string::npos)
            {
                //skip the following lines up to placeholder end
                while(std::getline(fin,line) && line.find(kPlaceholder_end)== std::string::npos);
                if(fin.eof()) //error we reached the end of file
                {
                    std::cerr<< "Placeholders not found " <<std::endl;
                    return false;
                }
                else
                    return true;

            }
            else
            {
                fout << line <<std::endl;
            }
        }

        //if we reached this point, placeholder haven't been found
        return false;
    }


    /**
     * @brief CopyTillEndReplacing copies the content of fin from the current position. Properly replaces module signature
     * @param fin
     * @param fout
     * @param isKernel
     * @param name
     */
    static void CopyTillEndReplacing(std::ifstream &fin, std::ofstream &fout, bool isKernel, std::string pattern, std::string name)
    {
        std::string line;
        while(std::getline(fin,line))
        {
             if(line.find("__kernel")!=std::string::npos)
             {
                 if(!isKernel)
                    line=std::regex_replace(line,std::regex("__kernel "),std::string(""));
                 line=std::regex_replace(line,std::regex(pattern),name);
             }
             fout << line<<std::endl;
        }
        fin.close();
    }


    /**
     * @brief CopyTillEnd copy the content of fin form the current position to end into fout.
     *  Fin is closed
     * @param fin
     * @param fout
     */
    static void CopyTillEnd(std::ifstream &fin, std::ofstream &fout)
    {
        std::string line;
        while(std::getline(fin,line))
        {
            fout << line<<std::endl;
        }
        fin.close();
    }



    static void CopyTillEnd(std::string file,std::ofstream &fout)
    {
        std::ifstream fin(file);
        if(!fin.is_open()){
            std::cerr << "Error in opening skeleton file for "<< file <<std::endl;
            return;
        }

        CopyTillEnd(fin,fout);
    }


    std::map<std::string, std::function<void(const GeneratorRoutine &r, unsigned int id, std::string)> > generation_functions_;

    std::map<std::string, std::function<void(const GeneratorHelper &h, unsigned int id, std::string)> > generation_helpers_;

    //definition of helper files
    static const std::map<std::string, std::string> k_helper_files_;

    static const std::string k_skeleton_folder_;            //the root directory containing all the skeletons
    static const std::string k_generated_file_header_;      //indicate that this file has been automatically generated


    //Macros defines
    //computation parameters and channel names defines (macro)
    static const std::string k_double_precision_define_;
    static const std::string k_width_define_;
    static const std::string k_ctile_rows_define_;
    static const std::string k_ctile_cols_define_;
    static const std::string k_tile_n_size_define_;
    static const std::string k_tile_m_size_define_;
    static const std::string k_mtile_size_define_;
    static const std::string k_kernel_name_define_ ;
    static const std::string k_channel_x_define_;
    static const std::string k_channel_y_define_;
    static const std::string k_channel_x_trans_define_;
    static const std::string k_channel_y_trans_define_;
    static const std::string k_channel_x_trsv_define_;
    static const std::string k_channel_matrix_A_define_;
    static const std::string k_channel_matrix_A2_define_;
    static const std::string k_channel_matrix_B_define_;
    static const std::string k_channel_matrix_B2_define_;
    static const std::string k_channel_scalar_out_define_;
    static const std::string k_channel_vector_out_define_;
    static const std::string k_channel_vector_out_x_define_;
    static const std::string k_channel_vector_out_y_define_;

    static const std::string k_channel_matrix_out_define_;
    static const std::string k_commons_define_;

    static const std::string k_stratix_10_platform_define_;
    static const std::string k_arria_10_platform_define_;
    static const std::string k_incx_define_;
    static const std::string k_incy_define_;
    static const std::string k_incw_define_;

    static const std::string k_channel_enable_define_;
};
#endif // MODULES_GENERATOR_HPP
