/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host API/Modules Generator Implementation: json parser
*/

#ifndef PARSER_HPP
#define PARSER_HPP
#include <vector>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <rapidjson/document.h>
#include <iostream>
#include <map>
#include <functional>
#include <unistd.h>
#include <set>
#include <algorithm>
#include "generator_routine.hpp"
#include "generator_helper.hpp"
#include "json_utils.hpp"
#include "generator_commons.hpp"


class JSONParser{

public:
    /**
     * @brief JSONParser
     * @param path_name path of the json file
     * @param tier: 1 for module generator, 2 for host layer generator
     */
    explicit JSONParser(const std::string path_name, unsigned int tier)
    {
        //read the json file
        std::ifstream fin (path_name.c_str());
        if(!fin)
            throw std::runtime_error("Problem in opening the json file");

        if(tier !=1 && tier !=2)
            throw std::runtime_error("Tier must be 1 or 2");
        std::vector<char> buffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        buffer.push_back('\0');
        fin.close();
        parse(buffer.data(),tier);


    }


    explicit JSONParser(const char *json, unsigned int tier)
    {
        //this constructure will be used for testing
        parse(json,tier);
        if(tier !=1 && tier !=2)
            throw std::runtime_error("Tier must be 1 or 2");

    }


    const std::vector<GeneratorRoutine> & getRoutines () const
    {
        return routines_;
    }

    const std::vector<GeneratorHelper> & getHelpers () const
    {
        return helpers_;
    }

private:
    void parse(const char *json,unsigned int tier)
    {
        if(document_.Parse(json).HasParseError() || ! document_.IsObject())
            throw std::runtime_error("Error in parsing file: malformed json");


        if(!document_.HasMember("routine"))
            throw std::runtime_error("Error in parsing file: no 'routine' object has been found!");

        //Load routine specification definition
        if(tier==2)
            loadRoutineDefinitionsTier2();
        else
            loadRoutineDefinitionsTier1();

        //get the target architecture (if specified, otherwise use the stratix 10)
        FBlasArchiteture target_arch;


        if(document_.HasMember("platform"))
        {
            rapidjson::Value &arch=document_["platform"];
            if(arch.IsString())
            {
                //get the architecture
                std::string architecture=arch.GetString();
                std::transform(architecture.begin(), architecture.end(), architecture.begin(), ::tolower);
                if(architecture == "stratix 10")
                    target_arch=FBLAS_STRATIX_10;
                else
                    if(architecture == "arria 10")
                        target_arch=FBLAS_ARRIA_10;
                    else
                    {
                        std::cerr << RED "Invalid target platform: unrecognized \""<<architecture<<"\". Stratix 10 will be used as target platform" RESET<<std::endl;
                        target_arch=FBLAS_STRATIX_10;
                    }


            }
            else
            {
                std::cerr << RED "Invalid \"platform\" property: it must be a string. Stratix 10 will be used as target platform" RESET<<std::endl;

            }
        }
        else
            target_arch=FBLAS_STRATIX_10;

        //Parse all the routines present in the file
        rapidjson::Value &routine=document_["routine"];

        if(routine.IsArray())
        {
            //parse all the routines in the array
            for(int i=0;i<routine.Size();i++)
                this->parseRoutine(routine[i],target_arch,tier);
        }
        else
            this->parseRoutine(routine,target_arch,tier);

        if(tier==1 && document_.HasMember("helper"))
        {
             rapidjson::Value &helper=document_["helper"];
             if(helper.IsArray())
             {
                 for(int i=0; i<helper.Size();i++)
                     this->parseHelpers(helper[i],target_arch);
             }
             else
                 this->parseHelpers(helper,target_arch);
        }

    }


    /**
     * @brief parseRoutine parse routines (for Modules or Host Generation) from the JSON file provided by the arguments
     * @param routine the JSON object indicating the routine
     *	Malformed routines will be dropped by displaying an error message
     */
    void parseRoutine(rapidjson::Value &routine, FblasArchitecture architecture, unsigned int tier)
    {


        //check for required routine elements (blas name, user name, precision)
        if(!JSONUtils::checkFieldAndType(routine,"blas","string"))
            return;
        std::string blas_name=std::string(routine["blas"].GetString());
        if(!JSONUtils::checkValidValue(blas_name,valid_blas_))
        {
            std::cerr << RED "Invalid routine name: unrecognized "<<blas_name<< RESET<<std::endl;
            return;
        }

        if(!JSONUtils::checkFieldAndType(routine,"user_name","string"))
        {
            std::cerr << RED "Invalid routine: missing \"user_name\" specification " RESET <<std::endl;
            return;
        }
        std::string user_name=std::string(routine["user_name"].GetString());

        if(!JSONUtils::checkFieldAndType(routine,"precision","string"))
            return;

        std::string precision=std::string(routine["precision"].GetString());

        if(precision != "single" && precision != "double")
        {
            std::cerr << RED "Invalid routine specification (routine: "<<user_name<<"): property \"precision\" must be \"single\" or \"double\""<<RESET<<std::endl;
            return;
        }


        //crete the routine and parse the rest of it
        GeneratorRoutine r(blas_name,user_name,precision,architecture,tier);

        //this part is in common for Tier 2 (Host Generator) and Tier 1 (Code generator)
        bool ok=true;
        if(twod_computational_tiled_routines_.find(blas_name)!=twod_computational_tiled_routines_.end())
        {
            r.set2DComputationalTile();
        }
        //check for the required field
        for(std::string req_f:required_parameters_[blas_name])
            ok&=ParseRoutineField(routine,r,req_f,true);

        //check for optional fields
        for(std::string opt_f:optional_parameters_[blas_name])
            ParseRoutineField(routine,r,opt_f,false);


        //check that (if defined) tile sizes are multiple of Width
        if(!r.has2DComputationalTile())
        {
            int width=(r.getWidth()!=0)? r.getWidth():k_default_width_;
            if(r.getTileNsize()!=0 && (r.getTileNsize() % width!=0))
            {
                std::cerr << RED "Invalid routine specification (routine: "<<user_name<<"): \"Tile N size\" must be a multiple of width (if width is not specified, by default it is "<<k_default_width_<<")"<< RESET<<std::endl;
                ok=false;
            }

            if(r.getTileMsize()!=0 && (r.getTileMsize() % width!=0))
            {
                std::cerr << RED "Invalid routine specification (routine: "<<user_name<<"): \"Tile M size\" must be a multiple of width (if width is not specified, by default it is "<<k_default_width_<<")" RESET<<std::endl;
                ok=false;
            }
        }
        else
        {
            int width_x=(r.getWidthX()!=0)? r.getWidthX():k_default_width_;
            int width_y=(r.getWidthY()!=0)? r.getWidthY():k_default_width_;
            if(r.getTileSize()!=0 && ((r.getTileSize() % width_x!=0) || (r.getTileSize() % width_y!=0)))
            {
                std::cerr << RED "Invalid routine specification (routine: "<<user_name<<"): \"Tile size\" must be a multiple of computational width (if width is not specified, by default it is "<<k_default_width_<<")" RESET<<std::endl;
                ok=false;
            }
        }

        if(tier==1)
        {
            //modules generator, we have other parameters
            for(std::string input_c:required_inputs_channels_[blas_name])
                ok&=parseChannel(routine,r,input_c,"input");
            for(std::string output_c:required_outputs_channels_[blas_name])
                ok&=parseChannel(routine,r,output_c,"output");
            //find the type
            if(!JSONUtils::checkFieldAndType(routine,"type","string"))
            {
                std::cerr << RED "Invalid routine specification (routine: "<<user_name<<"): missing type definition (kernel/function)"<< RESET<<std::endl;
                ok=false;
            }
            else
            {
                std::string type=routine["type"].GetString();
                if(type=="kernel")
                    r.setIsKernel(true);
                else
                    if(type=="function")
                        r.setIsKernel(false);
                    else
                    {
                        std::cerr << RED "Invalid routine specification (routine: "<<user_name<<"): property \"type\" must be \"function\" or \"kernel\""<<RESET<<std::endl;
                        ok=false;
                    }
            }

        }


        if(ok)
        {
            this->routines_.push_back(r);
        }

    }

    /**
     * @brief ParseRoutineField helper function to parse a required or optional field for a routine
     *	    Please note: allowed fields are limited and we now their types
     * @return true if the filed has been correctly parsed, false otherwise
     */
    bool ParseRoutineField(rapidjson::Value &routine, GeneratorRoutine &r,std::string field_name, bool mandatory)
    {
        if(field_name=="trans") //this version is used for level 2
        {
            //trans must be a string
            if(JSONUtils::checkFieldAndType(routine,field_name,"string"))
            {
                std::string trans_value=routine["trans"].GetString();
                if(trans_value=="N" || trans_value =="n")
                    r.setTransposeA(FBLAS_NO_TRANSPOSED);
                else
                    if(trans_value=="T" || trans_value =="t")
                        r.setTransposeA(FBLAS_TRANSPOSED);
                    else
                    {

                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"trans\" must have value 'N' or 'T'" RESET <<std::endl;
                        return false;
                    }
                return true;
            }
            else
            {
                if(mandatory)
                    std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): missing property \"trans\". It must be defined with value 'N' or 'T'" RESET <<std::endl;
                return false;
            }
        }
        //for level 3 we have to specify witch of the two input matrix is transposed
        if(field_name=="transa") //this version is used for level 2
        {
            //trans must be a string
            if(JSONUtils::checkFieldAndType(routine,field_name,"string"))
            {
                std::string trans_value=routine["transa"].GetString();
                if(trans_value=="N" || trans_value =="n")
                    r.setTransposeA(FBLAS_NO_TRANSPOSED);
                else
                    if(trans_value=="T" || trans_value =="t")
                        r.setTransposeA(FBLAS_TRANSPOSED);
                    else
                    {

                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"transa\" must have value 'N' or 'T'" RESET  <<std::endl;
                        return false;
                    }
                return true;
            }
            else
            {
                if(mandatory)
                    std::cerr << RED "Invalid routine specification: missing property \"transa\". It must be defined with value 'N' or 'T'" RESET <<std::endl;
                return false;
            }
        }
        if(field_name=="transb") //this version is used for level 2
        {
            //trans must be a string
            if(JSONUtils::checkFieldAndType(routine,field_name,"string"))
            {
                std::string trans_value=routine["transb"].GetString();
                if(trans_value=="N" || trans_value =="n")
                    r.setTransposeB(FBLAS_NO_TRANSPOSED);
                else
                    if(trans_value=="T" || trans_value =="t")
                        r.setTransposeB(FBLAS_TRANSPOSED);
                    else
                    {

                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"transb\" must have value 'N' or 'T'" RESET <<std::endl;
                        return false;
                    }
                return true;
            }
            else
            {
                if(mandatory)
                    std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): missing property \"transb\". It must be defined with value 'N' or 'T'" RESET <<std::endl;
                return false;
            }
        }

        if(field_name=="uplo")
        {
            //trans must be a string
            if(JSONUtils::checkFieldAndType(routine,field_name,"string"))
            {
                std::string uplo=routine["uplo"].GetString();
                if(uplo=="U" || uplo =="u")
                    r.setUplo(FBLAS_UPPER);
                else
                    if(uplo=="L" || uplo =="l")
                        r.setUplo(FBLAS_LOWER);
                    else
                    {

                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"uplo\" must have value 'U' or 'L'" RESET <<std::endl;
                        return false;
                    }
                return true;
            }
            else
            {
                if(mandatory)
                    std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): missing property \"uplo\". It must be defined with value 'N' or 'T'" RESET <<std::endl;
                return false;
            }
        }
        if(field_name=="side")
        {
            //trans must be a string
            if(JSONUtils::checkFieldAndType(routine,field_name,"string"))
            {
                std::string side=routine["side"].GetString();
                if(side=="L" || side =="l")
                    r.setSide(FBLAS_LEFT);
                else
                    if(side=="R" || side =="r")
                        r.setSide(FBLAS_RIGHT);
                    else
                    {

                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"side\" must have value 'L' or 'R'" RESET <<std::endl;
                        return false;
                    }
                return true;
            }
            else
            {
                if(mandatory)
                    std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): missing property \"side\". It must be defined with value 'L' or 'R'" RESET <<std::endl;
                return false;
            }
        }
        if(field_name=="order"){
            if(JSONUtils::checkFieldAndType(routine,field_name,"string"))
            {
                std::string order_value=routine["order"].GetString();
                if(order_value=="RowMajor" || order_value =="rowmajor")
                    r.setOrder(FBLAS_ROW_MAJOR);
                else//currently only RowMajor is supported
                    /*if(order_value=="ColumnMajor" || order_value =="columnmajor")
                        r.setOrder(FBLAS_COL_MAJOR);
                    else*/
                {
                    std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"order\" must have value 'RowMajor'" RESET  <<std::endl;
                    return false;
                }
                return true;
            }
            else{
                if(mandatory)
                    std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): missing property \"order\". It must be defined with value 'RowMajor'" RESET <<std::endl;
                return false;
            }
        }

        if(field_name=="width")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setWidth(routine["width"].GetUint());
                return true;
            }
            else
                return false;
        }

        if(field_name=="computational width x")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setWidthX(routine["computational width x"].GetUint());
                return true;
            }
            else
                return false;
        }
        if(field_name=="computational width y")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setWidthY(routine["computational width y"].GetUint());
                return true;
            }
            else
                return false;
        }

        if(field_name=="tile size")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setTileSize(routine["tile size"].GetUint());
                return true;
            }
            else
                return false;
        }

        if(field_name=="tile N size")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setTileNsize(routine["tile N size"].GetUint());
                return true;
            }
            else
                return false;
        }
        if(field_name=="tile M size")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setTileMsize(routine["tile M size"].GetUint());
                return true;
            }
            else
                return false;
        }

        if(field_name=="incx")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setIncx(routine["incx"].GetInt());
                return true;
            }
            else
                return false;
        }
        if(field_name=="incy")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setIncy(routine["incy"].GetInt());
                return true;
            }
            else
                return false;
        }

        if(field_name=="lda")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setLda(routine["lda"].GetUint());
                return true;
            }
            else
                return false;
        }

        if(field_name=="ldb")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"number"))
            {
                r.setLdb(routine["ldb"].GetUint());
                return true;
            }
            else
                return false;
        }
        //see if there are tile/elements order specification
        if(field_name=="A tiles order")
        {
            if(JSONUtils::checkFieldAndType(routine,"A tiles order","string"))
            {
                std::string order=std::string(routine["A tiles order"].GetString());
                if(order=="row")
                    r.setTilesARowStreamed(true);
                else
                    if(order=="column")
                        r.setTilesARowStreamed(false);
                    else
                    {
                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"A tiles order\" must be \"row\" or \"column\""<<RESET<<std::endl;
                        return false;
                    }
                return true;

            }
        }
        if(field_name=="A elements order")
        {
            if(JSONUtils::checkFieldAndType(routine,"A elements order","string"))
            {
                std::string order=std::string(routine["A elements order"].GetString());
                if(order=="row")
                    r.setElementsARowStreamed(true);
                else
                    if(order=="column")
                        r.setElementsARowStreamed(false);
                    else
                    {
                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"A elements order\" must be \"row\" or \"column\""<<RESET<<std::endl;
                        return false;
                    }
                return true;
            }
        }

        if(field_name=="B tiles order")
        {
            if(JSONUtils::checkFieldAndType(routine,"B tiles order","string"))
            {
                std::string order=std::string(routine["B tiles order"].GetString());
                if(order=="row")
                    r.setTilesBRowStreamed(true);
                else
                    if(order=="column")
                        r.setTilesBRowStreamed(false);
                    else
                    {
                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"B tiles order\" must be \"row\" or \"column\""<<RESET<<std::endl;
                        return false;
                    }
                return true;
            }
        }
        if(field_name=="B elements order")
        {
            if(JSONUtils::checkFieldAndType(routine,"B elements order","string"))
            {
                std::string order=std::string(routine["B elements order"].GetString());
                if(order=="row")
                    r.setElementsBRowStreamed(true);
                else
                    if(order=="column")
                        r.setElementsBRowStreamed(false);
                    else
                    {
                        std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): property \"B elements order\" must be \"row\" or \"column\""<<RESET<<std::endl;
                        return false;
                    }
                return true;
            }
        }

        if(field_name=="systolic")
        {
            if(JSONUtils::checkFieldAndType(routine,field_name,"bool"))
            {
                r.setSystolic(routine["systolic"].GetBool());
                return true;
            }
            else
                return false;

        }
        //not found
        return false;
    }

    bool parseChannel(rapidjson::Value &routine, GeneratorRoutine &r,std::string channel_name, std::string direction)
    {
        if(!JSONUtils::checkField(routine,channel_name))
        {
            std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): missing "<< direction<<" channel: "<<channel_name<<"" RESET <<std::endl;
            return false;
        }
        if(!r.addChannel(channel_name,routine[channel_name.c_str()].GetString()))
        {
            std::cerr << RED "Invalid routine specification (routine: "<<r.getUserName()<<"): already defined "<< direction<<" channel: "<<channel_name<<"" RESET <<std::endl;
            return false;
        }
        return true;
    }


    void parseHelpers(rapidjson::Value &helper, FblasArchitecture architecture)
    {

        //check for required routine elements (blas name, user name, precision)
        if(!JSONUtils::checkFieldAndType(helper,"type","string"))
        {
           std::cerr << RED "Invalid helper: missing type "<<RESET<<std::endl;
           return;
        }
        std::string type=std::string(helper["type"].GetString());
        if(!JSONUtils::checkValidValue(type,valid_helpers_))
        {
            std::cerr << RED "Invalid helper type: unrecognized "<<type<< RESET<<std::endl;
            return;
        }

        if(!JSONUtils::checkFieldAndType(helper,"user_name","string"))
        {
            std::cerr << RED "Invalid helper: missing \"user_name\" specification " RESET <<std::endl;
            return;
        }
        std::string user_name=std::string(helper["user_name"].GetString());

        if(!JSONUtils::checkFieldAndType(helper,"precision","string"))
        {
            std::cerr << RED "Invalid helper: missing \"precision\" specification " RESET <<std::endl;
            return;
        }

        std::string precision=std::string(helper["precision"].GetString());

        if(precision != "single" && precision != "double")
        {
            std::cerr << RED "Invalid helper specification (routine: "<<user_name<<"): property \"precision\" must be \"single\" or \"double\""<<RESET<<std::endl;
            return;
        }

        if(!JSONUtils::checkFieldAndType(helper,"channel_name","string"))
        {
            std::cerr << RED "Invalid helper: missing \"channel_name\" specification " RESET <<std::endl;
            return;
        }
        std::string channel_name=std::string(helper["channel_name"].GetString());

        //crete the routine and parse the rest of it
        GeneratorHelper h(type,user_name,precision,channel_name);

        if(JSONUtils::checkFieldAndType(helper,"width","number"))
        {
            h.setWidth(helper["width"].GetUint());
        }

        //if needed see if there is stride
        if(type=="read vector x" || type == "read vector y" || type=="write vector")
        {
            if(JSONUtils::checkFieldAndType(helper,"stride","number"))
                h.setStride(helper["stride"].GetInt());
        }

        //if it is a matrix, search for tile/elements order
        if(type == "read matrix" || type == "write matrix")
        {
            //mandatory: indication of tile sizes
            if(JSONUtils::checkFieldAndType(helper,"tile N size","number"))
                h.setTileNsize(helper["tile N size"].GetUint());
            else
            {
                 std::cerr << RED "Invalid helper: missing \"tile N size\" specification " RESET <<std::endl;
                 return ;
            }


            if(JSONUtils::checkFieldAndType(helper,"tile M size","number"))
                h.setTileMsize(helper["tile M size"].GetUint());
            else
            {
                std::cerr << RED "Invalid helper: missing \"tile M size\" specification " RESET <<std::endl;
                return ;
            }

            if(JSONUtils::checkFieldAndType(helper,"tiles order","string"))
            {
                std::string order=helper["tiles order"].GetString();
                if(order=="row")
                    h.setTilesRowStreamed(true);
                else
                    if(order=="column")
                        h.setTilesRowStreamed(false);
                    else
                        std::cerr << RED "Helper: property \"tiles order\" must be 'row' or 'column'. Assumed row. " RESET <<std::endl;
            }
            if(JSONUtils::checkFieldAndType(helper,"elements order","string"))
            {
                std::string order=helper["elements order"].GetString();
                if(order=="row")
                    h.setElementsRowStreamed(true);
                else
                    if(order=="column")
                        h.setElementsRowStreamed(false);
                    else
                        std::cerr << RED "Helper: property \"elements order\" must be 'row' or 'column'. Assumed row. " RESET <<std::endl;
            }
        }

        this->helpers_.push_back(h);

    }


    /**
     * Loads the routine definitions for Tier 1
     *
     * @note for the moment being is not used
     */
    void loadRoutineDefinitionsTier1()
    {
        //read the json file
        std::ifstream fin (routines_definitions_file_tier_1.c_str());

        if(!fin)
            throw std::runtime_error("Problem in opening the routine definitions file");

        std::vector<char> buffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        buffer.push_back('\0');
        fin.close();
        if(routine_defs_.Parse(buffer.data()).HasParseError() || ! routine_defs_.IsObject())
            throw std::runtime_error("Error in parsing routine definitions file");
        if(!routine_defs_.HasMember("routine"))
            throw std::runtime_error("Error in parsing routine definitions file: no routine object has been found!");

        //fill the map routine ->parserfunction
        //fillParsingFunctions();
        //Parse all the routines present in the file
        rapidjson::Value &routine=routine_defs_["routine"];
        if(routine.IsArray())
        {
            //parse all the routines definitions in the array
            for(int i=0;i<routine.Size();i++)
            {
                std::set<std::string> req_channels;
                std::set<std::string> required_parameters;
                std::set<std::string> optional_parameters;
                //check it has name and required_input fields
                if(!JSONUtils::checkField(routine[i],"name") || !JSONUtils::checkType(routine[i],"name","string"))
                {
                    std::cerr << RED "Internal error: Invalid routine specification in specification file" RESET<<std::endl;
                    return;
                }
                if(!JSONUtils::checkField(routine[i],"required_inputs") || !JSONUtils::checkType(routine[i],"required_inputs","array"))
                {
                    std::cerr << RED "Internal error: Invalid routine specification in specification file. Missing inputs." RESET<<std::endl;
                    return;
                }
                if(!JSONUtils::checkField(routine[i],"required_outputs") || !JSONUtils::checkType(routine[i],"required_outputs","array"))
                {
                    std::cerr << RED "Internal error: Invalid routine specification in specification file. Missing outputs." RESET<<std::endl;
                    return;
                }
                valid_blas_.insert(routine[i]["name"].GetString());
                //get the list of required inputs and outputs
                for(int j=0;j<routine[i]["required_inputs"].Size();j++)
                    req_channels.insert(routine[i]["required_inputs"][j].GetString());
                required_inputs_channels_[routine[i]["name"].GetString()]=req_channels;
                req_channels.clear();
                for(int j=0;j<routine[i]["required_outputs"].Size();j++)
                    req_channels.insert(routine[i]["required_outputs"][j].GetString());
                required_outputs_channels_[routine[i]["name"].GetString()]=req_channels;

                //required and optional parametrs
                if(JSONUtils::checkField(routine[i],"required_parameters"))
                {
                    if(!JSONUtils::checkType(routine[i],"required_parameters","array"))
                    {
                        std::cerr << RED "Internal Error: Invalid routine specification in specification file. Missing parameters. " RESET <<std::endl;
                    }
                    else
                    {
                        for(int j=0;j<routine[i]["required_parameters"].Size();j++)
                        {
                            required_parameters.insert(routine[i]["required_parameters"][j].GetString());
                            //TODO: check for admissible parameter?

                        }
                        required_parameters_[routine[i]["name"].GetString()]=required_parameters;
                    }
                }

                //check if there optional parameters
                if(JSONUtils::checkField(routine[i],"optional_parameters"))
                {
                    if(!JSONUtils::checkType(routine[i],"optional_parameters","array"))
                        std::cerr << RED "Internal Error: Invalid routine specification in specification file. Optional parameters malformed for " <<routine[i]["name"].GetString() <<RESET<<std::endl;
                    else
                    {
                        for(int j=0;j<routine[i]["optional_parameters"].Size();j++)
                        {
                            optional_parameters.insert(routine[i]["optional_parameters"][j].GetString());
                        }
                        optional_parameters_[routine[i]["name"].GetString()]=optional_parameters;
                    }
                }

            }
        }
        else{
            throw std::runtime_error("Error in parsing routine definitions file: routine must be a an array!");
        }



    }

    /*
     * Loads the routine definitions for Tier 2
     */
    void loadRoutineDefinitionsTier2()
    {
        //read the json file
        std::ifstream fin (routines_definitions_file_tier_2.c_str());

        if(!fin)
            throw std::runtime_error("Problem in opening the routine definitions file");

        std::vector<char> buffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
        buffer.push_back('\0');
        fin.close();

        if(routine_defs_.Parse(buffer.data()).HasParseError() || ! routine_defs_.IsObject())
            throw std::runtime_error("Error in parsing routine definitions file");
        if(!routine_defs_.HasMember("routine"))
            throw std::runtime_error("Error in parsing routine definitions file: no routine object has been found!");

        //Parse all the routines present in the file
        rapidjson::Value &routine=routine_defs_["routine"];
        if(routine.IsArray())
        {
            //parse all the routines definitions in the array
            for(int i=0;i<routine.Size();i++)
            {
                //check it has name and required_parameters fields
                std::set<std::string> required_parameters;
                std::set<std::string> optional_parameters;

                if(!JSONUtils::checkField(routine[i],"name") || !JSONUtils::checkType(routine[i],"name","string"))
                {
                    std::cerr << RED "Internal Error: Invalid routine specification in specification file" RESET<<std::endl;
                    return;
                }
                valid_blas_.insert(routine[i]["name"].GetString());

                if(JSONUtils::checkField(routine[i],"required_parameters"))
                {
                    if(!JSONUtils::checkType(routine[i],"required_parameters","array"))
                    {
                        std::cerr << RED "Internal Error: Invalid routine specification in specification file. Missing parameters. " RESET <<std::endl;
                    }
                    else
                    {
                        for(int j=0;j<routine[i]["required_parameters"].Size();j++)
                        {
                            required_parameters.insert(routine[i]["required_parameters"][j].GetString());
                            //TODO: check for admissible parameter?

                        }
                        required_parameters_[routine[i]["name"].GetString()]=required_parameters;
                    }
                }

                //check if there optional parameters
                if(JSONUtils::checkField(routine[i],"optional_parameters"))
                {
                    if(!JSONUtils::checkType(routine[i],"optional_parameters","array"))
                        std::cerr << RED "Internal Error: Invalid routine specification in specification file. Optional parameters malformed for " <<routine[i]["name"].GetString() <<RESET<<std::endl;
                    else
                    {
                        for(int j=0;j<routine[i]["optional_parameters"].Size();j++)
                        {
                            optional_parameters.insert(routine[i]["optional_parameters"][j].GetString());
                        }
                        optional_parameters_[routine[i]["name"].GetString()]=optional_parameters;
                    }
                }


            }
        }
        else{
            throw std::runtime_error("Error in parsing routine definitions file: routine must be a an array!");
        }



    }


    //set of valid blas
    std::set<std::string> valid_blas_ ;
    //set of required input/output channel for each (streaming) blas (TIER 1)
    std::map<std::string, std::set<std::string>> required_inputs_channels_;
    std::map<std::string, std::set<std::string>> required_outputs_channels_;
    //set of required parameters for each BLAS routine (TIER 2/2)
    std::map<std::string, std::set<std::string>> required_parameters_;
    //set of optional parameters for each BLAS routine (TIER 2/1)
    std::map<std::string, std::set<std::string>> optional_parameters_;
    //set of routines that have a 2D computational tile (mainly level 3, TIER 2)
    std::set<std::string> twod_computational_tiled_routines_={"gemm", "syrk", "syr2k"};

    const std::string routines_definitions_file_tier_1=std::string(BASE_FOLDER)+std::string("/include/generator/routines_definitions_tier_1.json");
    const std::string routines_definitions_file_tier_2=std::string(BASE_FOLDER)+std::string("/include/generator/routines_definitions_tier_2.json");
    rapidjson::Document document_;
    rapidjson::Document routine_defs_;

    //routines vector
    std::vector<GeneratorRoutine> routines_;

    //helpers vector
    std::vector<GeneratorHelper> helpers_;
    //set of valid helpers
    std::set<std::string> valid_helpers_={"read vector x", "read vector", "read vector y", "write scalar", "write vector", "read matrix" ,"write matrix"};
    //debug print flags
    bool verbose=false;
    bool very_verbose=true;

};
#endif // PARSER_HPP
