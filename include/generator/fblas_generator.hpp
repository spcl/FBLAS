/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host API Implementation: host code generator
*/

#ifndef BLAS_GENERATOR_HPP
#define BLAS_GENERATOR_HPP
#include <vector>
#include <map>
#include <functional>
#include <fstream>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include "generator_routine.hpp"
#include "generator_commons.hpp"

/**
 * @brief The FBlasGenerator class implements the FBlas code generator.
 *  Given a set of (parsed) routines it will generate the source code suitable for compilation
 */
class FBlasGenerator {
public:
    FBlasGenerator()
    {
        fillGenerationFunctions();
    }


    void GenerateCode(const std::vector<GeneratorRoutine> &routines, std::string output_dir)
    {
        if(routines.size()==0)
        {
            std::cout << RED "No routines to generate" RESET <<std::endl;
            return;
        }

        //std::cout << "Generating code for " <<routines.size()<<" routines"<<std::endl;
        rapidjson::StringBuffer s;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> json_writer(s);
        json_writer.StartObject();
        json_writer.Key("routine");
        json_writer.StartArray();

        //dobbiamo avere un insieme di channel descriptors
        for(int i=0; i< routines.size();i++)
        {
            std::cout << "Generating: "<<routines[i].getUserName()<<std::endl;
            //For each routine the generated code will looks like:
            //1) Defines (names, precision, channels, width, ...)
            //2) Main computational kernel
            //3) Helper kernels (read/write from /to memory)
            if(generation_functions_.find(routines[i].getBlasName())!=generation_functions_.end())
                generation_functions_[routines[i].getBlasName()](routines[i],i,output_dir,json_writer);
            else
                std::cout << "Generation function not present " << routines[i].getBlasName()<<std::endl;
        }

        //close and writes the json
        json_writer.EndArray();
        json_writer.EndObject();
        std::ofstream fout(output_dir+"generated_routines.json");
        if(!fout.is_open()){
            std::cerr << "Error in opening output file generated_routines.json (file path: "<<output_dir<<")"<<std::endl;
            return;
        }
        fout << s.GetString();
        fout.close();

    }


private:

    //Level 1
    static void Level1Dot(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Axpy(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Scal(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Asum(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Iamax(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Nrm2(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Rot(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Rotm(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Rotg(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Rotmg(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Copy(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level1Swap(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);

    //Level 2
    static void Level2Gemv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level2Ger(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level2Syr(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level2Syr2(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level2Trsv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level2Trmv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level2Symv(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);


    //Level 3
    static void Level3Gemm(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level3Syrk(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level3Syr2k(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);
    static void Level3Trsm(const GeneratorRoutine &r, unsigned int id, std::string output_folder, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer);



    //Generation functions
    void fillGenerationFunctions()
    {
        //	    parsing_functions_.insert(std::make_pair("sdot",BlasParser::Level1Sdot));
        generation_functions_["axpy"]=FBlasGenerator::Level1Axpy;
        generation_functions_["dot"]=FBlasGenerator::Level1Dot;
        generation_functions_["scal"]=FBlasGenerator::Level1Scal;
        generation_functions_["asum"]=FBlasGenerator::Level1Asum;
        generation_functions_["nrm2"]=FBlasGenerator::Level1Nrm2;
        generation_functions_["iamax"]=FBlasGenerator::Level1Iamax;
        generation_functions_["rot"]=FBlasGenerator::Level1Rot;
        generation_functions_["rotm"]=FBlasGenerator::Level1Rotm;
        generation_functions_["rotg"]=FBlasGenerator::Level1Rotg;
        generation_functions_["rotmg"]=FBlasGenerator::Level1Rotmg;
        generation_functions_["copy"]=FBlasGenerator::Level1Copy;
        generation_functions_["swap"]=FBlasGenerator::Level1Swap;


        //level 2
        generation_functions_["gemv"]=FBlasGenerator::Level2Gemv;
        generation_functions_["ger"]=FBlasGenerator::Level2Ger;
        generation_functions_["syr"]=FBlasGenerator::Level2Syr;
        generation_functions_["syr2"]=FBlasGenerator::Level2Syr2;
        generation_functions_["trsv"]=FBlasGenerator::Level2Trsv;
        generation_functions_["trmv"]=FBlasGenerator::Level2Trmv;
        generation_functions_["symv"]=FBlasGenerator::Level2Symv;

        //Level 3
        generation_functions_["trsm"]=FBlasGenerator::Level3Trsm;
        generation_functions_["syrk"]=FBlasGenerator::Level3Syrk;
        generation_functions_["syr2k"]=FBlasGenerator::Level3Syr2k;
        generation_functions_["gemm"]=FBlasGenerator::Level3Gemm;



    }

    /**
     * @brief addCommons starts to fill up the generating file and the routine genereated json item
     *      with characteristics that are common to all the routines (name, user_name, precision, width)
     * @param blas_name
     * @param r
     * @param json_writer
     * @param fout
     */
    static void addCommons(std::string blas_name,const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        json_writer.StartObject();
        json_writer.Key(k_json_field_blas_name);
        json_writer.String(blas_name.c_str());
        json_writer.Key(k_json_field_user_name);
        json_writer.String(r.getUserName().c_str());
        //precision
        json_writer.Key(k_json_field_precision);

        if(r.isDoublePrecision())
        {
            fout << k_double_precision_define_ <<std::endl;
            json_writer.String("double");
        }
        else
            json_writer.String("single");

        //width: how to setup this information depends on the type of routine
        if(!r.has2DComputationalTile())
        {
            json_writer.Key(k_json_field_width);

            fout << k_width_define_;
            unsigned int width=(r.getWidth()!=0)?r.getWidth():k_default_width_;
            fout << width <<std::endl;
            json_writer.Uint(width);
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
            json_writer.Key(k_json_field_width_x);
            json_writer.Uint(width_x);
            json_writer.Key(k_json_field_width_y);
            json_writer.Uint(width_y);
        }

    }

    /**
     * @brief addTileN add the informations about the Tile Size referring to N
     *  to both the generating kernel and routine json definition
     */
    static void addTileN(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        json_writer.Key(k_json_field_tile_n_size);
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
        json_writer.Uint(size);
    }

    /**
     * @brief addTileM add the informations about the Tile Size referring to M
     *  to both the generating kernel and routine json definition
     */
    static void addTileM(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        json_writer.Key(k_json_field_tile_m_size);
        fout << k_tile_m_size_define_;
        unsigned int size= (r.getTileMsize()!=0)? r.getTileMsize() : k_default_tiling_;
        if(r.getWidth() != 0 && (size%r.getWidth() != 0 || size<r.getWidth()))
        {
            size = r.getWidth() * 4 ;
            std::cout << "Tile M size for routine " << r.getUserName() << " was not a multiple of the width. Set at: "<<size<<std::endl;
        }
        fout << size <<std::endl;
        json_writer.Uint(size);
    }

    /**
     * @brief addTileMadd the informations about the Tile Size referring in the case of 2D computational tiled routines.
     *  The info is contained in both the generated kernel and routine json definition
     */
    static void addTile(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        json_writer.Key(k_json_field_mtile_size);
        fout << k_mtile_size_define_;
        unsigned int size= (r.getTileSize()!=0)? r.getTileSize() : k_default_tiling_;
        //check that it is a multiple of width x and y
        if((r.getWidthX() != 0 && size%r.getWidthX() != 0) || (r.getWidthY() != 0 && size%r.getWidthY() != 0))
        {
            size = ((r.getWidthX()!=0)?r.getWidthX():k_default_2d_width_) * ((r.getWidthY()!=0)?r.getWidthY():k_default_2d_width_);
            std::cout << "Tile size for routine " << r.getUserName() << " was not a multiple of the computational width x and y. Set at: "<<size<<std::endl;
        }
        fout << size <<std::endl;
        json_writer.Uint(size);
    }

    /**
     * @brief addIncX add the information about incx to both generated kernel and routine json
     */
    static void addIncX(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        fout << k_incx_define_;
        fout << r.getIncx()<<std::endl;
        json_writer.Key(k_json_field_incx);
        json_writer.Int(r.getIncx());
    }

    /**
     * @brief addIncY add the information about incy to both generated kernel and routine json
     */
    static void addIncY(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        fout << k_incy_define_;
        fout << r.getIncy()<<std::endl;
        json_writer.Key(k_json_field_incy);
        json_writer.Int(r.getIncy());
    }
    /**
     * @brief addIncW add the information about a generic IncW, access stride to a vector
     *      The stride is passed as argument. NOTE: this add only ad define
     */
    static void addDefineIncW(const GeneratorRoutine &r, int incw, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        fout << k_incw_define_;
        fout << incw <<std::endl;
        //json_writer.Key(k_json_field_incy);
        //json_writer.Int(r.getIncy());
    }

    /**
     * @brief addOrder add the information about the Order (rowMajor/ColumnMajor)
     */
    static void addOrder(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
    {
        json_writer.Key(k_json_field_order);
        if(r.getOrder()==FBLAS_ROW_MAJOR)
            json_writer.String("RowMajor");
        else
            json_writer.String("ColumnMajor");
    }

    /**
     * @brief addTranspose add information about the characterstic of the Matrix (matrix A, transposed or non transposed)
     */
    static void addTranspose(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
    {
        json_writer.Key(k_json_field_transA);
        if(r.getTransposeA()==FBLAS_NO_TRANSPOSED)
            json_writer.String("N");
        else
            json_writer.String("T");

    }

    /**
     * @brief addTranspose add information about the characterstic of the Matrix B if present (transposed or non transposed)
     */
    static void addTransposeB(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
    {
        json_writer.Key(k_json_field_transB);
        if(r.getTransposeB()==FBLAS_NO_TRANSPOSED)
            json_writer.String("N");
        else
            json_writer.String("T");

    }

    /**
     * @brief addUplo add information about the characterstic of the input Matrix (upper/lower)
     */
    static void addUpLo(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
    {
        json_writer.Key(k_json_field_uplo);
        if(r.getUplo()==FBLAS_LOWER)
            json_writer.String("L");
        else
            json_writer.String("U");
    }

    /**
     * @brief addside add information about the characterstic of the input Matrix (upper/lower)
     */
    static void addSide(const GeneratorRoutine &r, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
    {
        json_writer.Key(k_json_field_side);
        if(r.getSide()==FBLAS_LEFT)
            json_writer.String("L");
        else
            json_writer.String("R");
    }





    /**
     * @brief addDefineAndItem add a given define and a given item, both of them with a given string as value
     */
    static void addDefineAndItem(std::string define_name, const char * field_name, std::string value, rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        fout << define_name<<value<<std::endl;
        json_writer.Key(field_name);
        json_writer.String(value.c_str());
    }

    /**
     * @brief closeRoutineItem closes the routine item
     */
    static void closeRoutineItem(rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer)
    {
        json_writer.EndObject();
    }

    /**
     * @brief addDefineAndItemHelperReadVectorX add defines and json field for the helper that reads vector x
     */
    static void addDefineAndItemHelperReadVectorX(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_read_vector_x_base_name_+std::to_string(id);
        addDefineAndItem(k_read_vector_x_name_define_,k_json_field_read_vector_x,name,json_writer,fout);
    }

    static void addDefineAndItemHelperReadVectorXTrans(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_read_vector_x_trans_base_name_+std::to_string(id);
        addDefineAndItem(k_read_vector_x_trans_name_define_,k_json_field_read_vector_x_trans,name,json_writer,fout);
    }

    static void addDefineAndItemHelperReadVectorXTrsv(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_read_vector_x_trsv_base_name_+std::to_string(id);
        addDefineAndItem(k_read_vector_x_trsv_name_define_,k_json_field_read_vector_x_trsv,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperReadVectorY add defines and json field for the helper that reads vector y
     */
    static void addDefineAndItemHelperReadVectorY(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_read_vector_y_base_name_+std::to_string(id);
        addDefineAndItem(k_read_vector_y_name_define_,k_json_field_read_vector_y,name,json_writer,fout);
    }

    static void addDefineAndItemHelperReadVectorYTrans(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_read_vector_y_trans_base_name_+std::to_string(id);
        addDefineAndItem(k_read_vector_y_trans_name_define_,k_json_field_read_vector_y_trans,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperReadMatrixA add define and json field for the read matrix A helper
     */
    static void addDefineAndItemHelperReadMatrixA(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=k_kernel_read_matrix_A_base_name_+std::to_string(id);
        addDefineAndItem(k_read_matrix_A_name_define_,k_json_field_read_matrix_A,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperReadMatrixA2 add define and json field for the read matrix A2 helper (used for syrk/syr2k)
     */
    static void addDefineAndItemHelperReadMatrixA2(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=k_kernel_read_matrix_A2_base_name_+std::to_string(id);
        addDefineAndItem(k_read_matrix_A2_name_define_,k_json_field_read_matrix_A2,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperReadMatrixB add define and json field for the read matrix B helper
     */
    static void addDefineAndItemHelperReadMatrixB(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=k_kernel_read_matrix_B_base_name_+std::to_string(id);
        addDefineAndItem(k_read_matrix_B_name_define_,k_json_field_read_matrix_B,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperReadMatrixB2 add define and json field for the read matrix B helper (used for syr2k)
     */
    static void addDefineAndItemHelperReadMatrixB2(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=k_kernel_read_matrix_B2_base_name_+std::to_string(id);
        addDefineAndItem(k_read_matrix_B2_name_define_,k_json_field_read_matrix_B2,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperWriteVector add define and json item for write vector helpers
     */
    static void addDefineAndItemHelperWriteVector(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_write_vector_base_name_+std::to_string(id);
        addDefineAndItem(k_write_vector_name_define_,k_json_field_write_vector,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperWriteVectorXY add define and json item for write two vectors helpers
     */
    static void addDefineAndItemHelperWriteVectorXY(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_write_vector_x_y_base_name_+std::to_string(id);
        addDefineAndItem(k_write_vector_x_y_name_define_,k_json_field_write_vector,name,json_writer,fout);
    }

    /**
     * @brief addDefineAndItemHelperWriteScalar
     */
    static void addDefineAndItemHelperWriteScalar(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_write_scalar_base_name_ + std::to_string(id);
        addDefineAndItem(k_write_scalar_name_define_,k_json_field_write_scalar,name,json_writer,fout);

    }

    static void addDefineAndItemHelperWriteMatrix(const unsigned int id,rapidjson::PrettyWriter<rapidjson::StringBuffer> &json_writer, std::ofstream &fout)
    {
        std::string name=FBlasGenerator::k_kernel_write_matrix_base_name_ + std::to_string(id);
        addDefineAndItem(k_write_matrix_name_define_,k_json_field_write_matrix,name,json_writer,fout);
    }

    /**
     * @brief addDefineKernelName add the kernel name to the generating opencl file
     */
    static void addDefineKernelName(const GeneratorRoutine &r, std::ofstream &fout)
    {
        fout << k_kernel_name_define_<<r.getUserName()<<std::endl;
    }

    /**
     * @brief addDefineChannelInVectorX add the define for input channel vector X
     */
    static void addDefineChannelInVectorX(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_vector_x_base_name_+std::to_string(id);
        fout << k_channel_x_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelInVectorXTrans for Syr and Syr2
     */
    static void addDefineChannelInVectorXTrans(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_vector_x_trans_base_name_+std::to_string(id);
        fout << k_channel_x_trans_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelInVectorXTrsv for trsv
     */
    static void addDefineChannelInVectorXTrsv(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_vector_x_trsv_base_name_+std::to_string(id);
        fout << k_channel_x_trsv_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelInVectorY add the define for input channel vector Y to the generating opencl kernel
     */
    static void addDefineChannelInVectorY(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_vector_y_base_name_+std::to_string(id);
        fout << k_channel_y_define_ << name<<std::endl;
    }

    static void addDefineChannelInVectorYTrans(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_vector_y_trans_base_name_+std::to_string(id);
        fout << k_channel_y_trans_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelInMatrixA add the define for input channel Matrix A to the generating opencl kernel
     */
    static void addDefineChannelInMatrixA(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_matrix_A_base_name_+std::to_string(id);
        fout << k_channel_matrix_A_define_ <<name<<std::endl;
    }

    /**
     * @brief addDefineChannelInMatrixA2 add the define for input channel Matrix A to the generating opencl kernel
     * (used for syrk/syr2k)
     */
    static void addDefineChannelInMatrixA2(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_matrix_A2_base_name_+std::to_string(id);
        fout << k_channel_matrix_A2_define_ <<name<<std::endl;
    }

    /**
     * @brief addDefineChannelInMatrixB add the define for input channel Matrix B to the generating opencl kernel
     */
    static void addDefineChannelInMatrixB(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_matrix_B_base_name_+std::to_string(id);
        fout << k_channel_matrix_B_define_ <<name<<std::endl;
    }
    /**
     * @brief addDefineChannelInMatrixB2 add the define for input channel Matrix B to the generating opencl kernel
     * (used for syr2k)
     */
    static void addDefineChannelInMatrixB2(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_in_matrix_B2_base_name_+std::to_string(id);
        fout << k_channel_matrix_B2_define_ <<name<<std::endl;
    }


    /**
     * @brief addDefineChannelOutScalar add define for channel out scalar
     */
    static void addDefineChannelOutScalar(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_out_scalar_base_name_+std::to_string(id);
        fout << k_channel_scalar_out_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelOutVector add define for channel out vector
     */
    static void addDefineChannelOutVector(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_out_vector_base_name_+std::to_string(id);
        fout << k_channel_vector_out_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelOutVectorXY add define for channel out for two vectors
     */
    static void addDefineChannelOutVectorXY(const unsigned int id, std::ofstream &fout)
    {
        std::string name=k_channel_out_vector_x_base_name_+std::to_string(id);
        fout << k_channel_vector_out_x_define_ << name<<std::endl;

        name=k_channel_out_vector_y_base_name_+std::to_string(id);
        fout << k_channel_vector_out_y_define_ << name<<std::endl;
    }

    /**
     * @brief addDefineChannelOutMatrix add define for channel out matrix
     */
    static void addDefineChannelOutMatrix(const unsigned int id,std::ofstream &fout)
    {
        std::string name=k_channel_out_matrix_base_name_+std::to_string(id);
        fout << k_channel_matrix_out_define_ << name<<std::endl;
    }

    /**
     * @brief CopyHeader copies the header of the file till the placeholder is found, skipping the
     *              first block of comments (License)
     * @param fin
     * @param fout
     * @return true if the placeholder have been found, false otherwise
     */
    static bool CopyHeader(std::ifstream &fin, std::ofstream &fout)
    {
        const std::string kPlaceholder_start="//FBLAS_PARAMETERS_START";
        const std::string kPlaceholder_end="//FBLAS_PARAMETERS_END";
        bool first_comment_block_skipped=false;
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

    /**
     * @brief CopyHelper copies the content of an helper into the file passed as argument
     *      Skips the first comment block
     * @param file path to the helper
     * @param fout
     */
    static void CopyHelper(std::string file,std::ofstream &fout)
    {
        std::ifstream fin(file);
        if(!fin.is_open()){
            std::cerr << "Error in opening skeleton file for "<< file <<std::endl;
            return;
        }
        std::string line;
        bool first_comment_block_skipped=false;
        while(std::getline(fin,line) && !first_comment_block_skipped)
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
        }
        if(!first_comment_block_skipped)
        {
            std::cerr << "Error in opening skeleton file for "<< file << " (not able to skip the first comment block)" <<std::endl;
            return;
        }
        fout << line <<std::endl;
        CopyTillEnd(fin,fout);
    }

    std::map<std::string, std::function<void(const GeneratorRoutine &r, unsigned int id, std::string, rapidjson::PrettyWriter<rapidjson::StringBuffer> &)> > generation_functions_;

    //definition of helper files
    static const std::map<std::string, std::string> k_helper_files_;

    //Constants

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

    //helper Kernel name defines (macro)
    static const std::string k_read_vector_x_name_define_;
    static const std::string k_read_vector_y_name_define_;
    static const std::string k_read_vector_x_trans_name_define_;
    static const std::string k_read_vector_y_trans_name_define_;
    static const std::string k_read_vector_x_trsv_name_define_;
    static const std::string k_read_matrix_A_name_define_;
    static const std::string k_read_matrix_A2_name_define_;
    static const std::string k_read_matrix_B_name_define_;
    static const std::string k_read_matrix_B2_name_define_;
    static const std::string k_write_scalar_name_define_;
    static const std::string k_write_vector_name_define_;
    static const std::string k_write_vector_x_y_name_define_;
    static const std::string k_write_matrix_name_define_;
    static const std::string k_incx_define_;
    static const std::string k_incy_define_;
    static const std::string k_incw_define_;

    //base names for channels and kernels
    static const std::string k_channel_in_vector_x_base_name_;
    static const std::string k_channel_in_vector_y_base_name_;
    static const std::string k_channel_in_vector_x_trans_base_name_;
    static const std::string k_channel_in_vector_y_trans_base_name_;
    static const std::string k_channel_in_vector_x_trsv_base_name_; //for trsv we need a special vector reader
    static const std::string k_channel_in_matrix_A_base_name_;
    static const std::string k_channel_in_matrix_A2_base_name_;
    static const std::string k_channel_in_matrix_B_base_name_;
    static const std::string k_channel_in_matrix_B2_base_name_;
    static const std::string k_channel_out_scalar_base_name_;
    static const std::string k_channel_out_vector_base_name_;
    static const std::string k_channel_out_vector_x_base_name_;
    static const std::string k_channel_out_vector_y_base_name_;
    static const std::string k_channel_out_matrix_base_name_;

    static const std::string k_kernel_read_vector_x_base_name_;
    static const std::string k_kernel_read_vector_y_base_name_;
    static const std::string k_kernel_read_vector_x_trans_base_name_;
    static const std::string k_kernel_read_vector_y_trans_base_name_;
    static const std::string k_kernel_read_vector_x_trsv_base_name_;
    static const std::string k_kernel_read_matrix_A_base_name_;
    static const std::string k_kernel_read_matrix_A2_base_name_;
    static const std::string k_kernel_read_matrix_B_base_name_;
    static const std::string k_kernel_read_matrix_B2_base_name_;
    static const std::string k_kernel_write_scalar_base_name_;
    static const std::string k_kernel_write_vector_base_name_;
    static const std::string k_kernel_write_vector_x_y_base_name_;
    static const std::string k_kernel_write_matrix_base_name_;

    //helpers filenames
    //this are very peculiar for each helper (e.g. read matrix by row, by col...)
    //while the previous ones where more generic
    static const std::string k_helper_read_vector_x_;
    static const std::string k_helper_read_vector_y_;
    static const std::string k_helper_read_vector_x_trans_low_;
    static const std::string k_helper_read_vector_y_trans_low_;
    static const std::string k_helper_read_vector_x_trans_upper_;
    static const std::string k_helper_read_vector_y_trans_upper_;
    static const std::string k_helper_read_vector_x_trsv_low_;
    static const std::string k_helper_read_vector_x_trsv_upper_;

    static const std::string k_helper_write_vector_;
    static const std::string k_helper_write_vector_x_y_;
    static const std::string k_helper_write_scalar_;
    static const std::string k_helper_write_integer_;
    static const std::string k_helper_read_matrix_rowstreamed_tile_row_;

    static const std::string k_helper_read_matrix_rowstreamed_tile_col_;
    static const std::string k_helper_read_lower_matrix_rowstreamed_tile_row_;
    static const std::string k_helper_read_upper_matrix_rowstreamed_tile_row_;
    static const std::string k_helper_read_upper_matrix_rowstreamed_tile_col_;
    static const std::string k_helper_read_reverse_upper_matrix_rowstreamed_tile_row_;
    static const std::string k_helper_read_reverse_lower_matrix_rowstreamed_tile_col_;
    static const std::string k_helper_write_matrix_rowstreamed_tile_row_;
    static const std::string k_helper_write_lower_matrix_rowstreamed_tile_row_;
    static const std::string k_helper_write_upper_matrix_rowstreamed_tile_row_;

    //level 3
    static const std::string k_helper_read_matrix_a_notrans_gemm_;
    static const std::string k_helper_read_matrix_a_notrans_syrk_;
    static const std::string k_helper_read_matrix_a_trans_syrk_;
    static const std::string k_helper_read_matrix_a2_trans_syrk_;
    static const std::string k_helper_read_matrix_a2_notrans_syrk_;
    static const std::string k_helper_read_matrix_a_trans_gemm_;
    static const std::string k_helper_read_matrix_b_notrans_gemm_;
    static const std::string k_helper_read_matrix_b_trans_gemm_;
    static const std::string k_helper_read_matrix_b_trans_syr2k_;
    static const std::string k_helper_read_matrix_b2_trans_syr2k_;
    static const std::string k_helper_read_matrix_b_notrans_syr2k_;
    static const std::string k_helper_read_matrix_b2_notrans_syr2k_;
    static const std::string k_helper_write_matrix_gemm_;
    static const std::string k_helper_write_lower_matrix_syrk_;
    static const std::string k_helper_write_upper_matrix_syrk_;




    //special helpers (e.g. for TRMV, ...)
    static const std::string k_helper_read_matrix_as_lower_rowstreamed_tile_row_;
    static const std::string k_helper_read_matrix_as_upper_rowstreamed_tile_row_;
    static const std::string k_helper_read_matrix_as_upper_rowstreamed_tile_col_;
    static const std::string k_helper_read_matrix_as_lower_rowstreamed_tile_col_;
    static const std::string k_helper_read_matrix_as_symmetric_lower_rowstreamed_tile_row_;
    static const std::string k_helper_read_matrix_as_symmetric_upper_rowstreamed_tile_col_;

};


#endif // BLAS_GENERATOR_HPP

