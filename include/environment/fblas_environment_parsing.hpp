/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host Api Implementation - Parse for the JSON file produced by the host_generator
*/


#ifndef FBLAS_ENVIRONMENT_PARSING_HPP
#define FBLAS_ENVIRONMENT_PARSING_HPP
#include "routine.hpp"
#include "../commons.hpp"
/**
  Contains the definition of parsing methods for the JSON file produced by the generator
*/

void FBLASEnvironment::parseJSON(std::string json_file)
{
    //Since the JSON is automatically generated and well-formed, we will skip correctness check
    //read the json file
    std::ifstream fin (json_file.c_str());
    if(!fin)
        throw std::runtime_error("Problem in opening the json file");

    std::vector<char> buffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    buffer.push_back('\0');
    fin.close();

    rapidjson::Document document;
    document.Parse(buffer.data());
    rapidjson::Value &routine=document["routine"];

    for(int i=0; i<routine.Size();i++)
    {
        this->parseRoutine(routine[i]);
    }


}

void FBLASEnvironment::parseRoutine(rapidjson::Value &routine)
{
    Routine r;
    //parse all the fields
    //the required ones (i.e. required for every routine)
    r.blas_name=routine[k_json_field_blas_name].GetString();
    r.user_name=routine[k_json_field_user_name].GetString();

    r.double_precision=(routine[k_json_field_precision].GetString()=="double");
    if(r.blas_name != "rotg" && r.blas_name != "rotmg")   //copy, rotg and rotmg are only particular case, which is realized just by using helpers
    {
        if(routine.HasMember(k_json_field_width))
            r.width=routine[k_json_field_width].GetUint();
        else
        {
            //2D computational tiling
            //it must have one of the two kind of width
            r.has2DComputationalTiling=true;
            r.width_x=routine[k_json_field_width_x].GetUint();
            r.width_y=routine[k_json_field_width_y].GetUint();
        }
    }



    //the not mandatory (i.e. order that is not needed for Level 1 routines)
    if(routine.HasMember(k_json_field_order))
        if(std::string(routine[k_json_field_order].GetString())=="RowMajor")
        {
            r.order=FBLAS_ROW_MAJOR;
        }
        else
            r.order=FBLAS_COL_MAJOR;

    if(routine.HasMember(k_json_field_transA))
        if(std::string(routine[k_json_field_transA].GetString())=="N")
            r.transA=FBLAS_NO_TRANSPOSED;
        else
            r.transA=FBLAS_TRANSPOSED;

    if(routine.HasMember(k_json_field_transB))
        if(std::string(routine[k_json_field_transB].GetString())=="N")
            r.transB=FBLAS_NO_TRANSPOSED;
        else
            r.transB=FBLAS_TRANSPOSED;

    if(routine.HasMember(k_json_field_uplo))
        if(std::string(routine[k_json_field_uplo].GetString())=="L")
            r.uplo=FBLAS_LOWER;
        else
            r.uplo=FBLAS_UPPER;

    if(routine.HasMember(k_json_field_side))
        if(std::string(routine[k_json_field_side].GetString())=="L")
            r.side=FBLAS_LEFT;
        else
            r.side=FBLAS_RIGHT;

    if(routine.HasMember(k_json_field_tile_n_size))
        r.tile_n_size=routine[k_json_field_tile_n_size].GetUint();

    if(routine.HasMember(k_json_field_tile_m_size))
        r.tile_m_size=routine[k_json_field_tile_m_size].GetUint();

    if(routine.HasMember(k_json_field_incx))
        r.incx=routine[k_json_field_incx].GetInt();

    if(routine.HasMember(k_json_field_incy))
        r.incy=routine[k_json_field_incy].GetInt();

    if(routine.HasMember(k_json_field_lda))
        r.lda=routine[k_json_field_lda].GetUint();

    if(routine.HasMember(k_json_field_ldb))
        r.ldb=routine[k_json_field_ldb].GetUint();

    if(routine.HasMember(k_json_field_systolic))
        r.systolic=routine[k_json_field_systolic].GetBool();


    //create the kernels and the command queues
    //kernels are created and inserted respecting a proper order which
    //reflect the one with which they are written in the JSON file from
    //the code generator (see implementation notes)
    //- the first kernel will be the computational one
    //- then there are the reading kernels
    //      - for matrices: in the order A and B, if present
    //      - for vector: in order x (and x transposed if present) and y (and y transposed if present)
    //- then the sink (if present, e.g. strsv does not have the sink)
    //In this part we will probe for any one of them

    if(r.blas_name != "copy" && r.blas_name != "rotg" && r.blas_name != "rotmg")   //copy, rotg and rotmg are only particular case, which is realized just by using helpers
        r.kernels_names.push_back(r.user_name);

    //if(routine.HasMember("read_matrix_A"))
    //    r.kernels_names.push_back(routine["read_matrix_A"].GetString());
    if(routine.HasMember(k_json_field_read_matrix_A))
       r.kernels_names.push_back(routine[k_json_field_read_matrix_A].GetString());
    if(routine.HasMember(k_json_field_read_matrix_A2))
       r.kernels_names.push_back(routine[k_json_field_read_matrix_A2].GetString());
    if(routine.HasMember(k_json_field_read_matrix_B))
       r.kernels_names.push_back(routine[k_json_field_read_matrix_B].GetString());
    if(routine.HasMember(k_json_field_read_matrix_B2))
       r.kernels_names.push_back(routine[k_json_field_read_matrix_B2].GetString());


    if(routine.HasMember(k_json_field_read_vector_x))
        r.kernels_names.push_back(routine[k_json_field_read_vector_x].GetString());
    if(routine.HasMember(k_json_field_read_vector_x_trans))
        r.kernels_names.push_back(routine[k_json_field_read_vector_x_trans].GetString());
    if(routine.HasMember(k_json_field_read_vector_x_trsv))
        r.kernels_names.push_back(routine[k_json_field_read_vector_x_trsv].GetString());

    if(routine.HasMember(k_json_field_read_vector_y))
        r.kernels_names.push_back(routine[k_json_field_read_vector_y].GetString());
    if(routine.HasMember(k_json_field_read_vector_y_trans))
        r.kernels_names.push_back(routine[k_json_field_read_vector_y_trans].GetString());
    if(routine.HasMember(k_json_field_write_scalar))
        r.kernels_names.push_back(routine[k_json_field_write_scalar].GetString());
    if(routine.HasMember(k_json_field_write_vector))
        r.kernels_names.push_back(routine[k_json_field_write_vector].GetString());
    if(routine.HasMember(k_json_field_write_matrix))
        r.kernels_names.push_back(routine[k_json_field_write_matrix].GetString());

    //create kernels and queueus
    IntelFPGAOCLUtils::createKernels(program_,r.kernels_names,r.kernels);
    IntelFPGAOCLUtils::createCommandQueues(context_,device_,r.queues,r.kernels_names.size());



    //move to map
    routines_[r.user_name]=std::move(r);



}


#endif // FBLAS_ENVIRONMENT_PARSING_HPP
