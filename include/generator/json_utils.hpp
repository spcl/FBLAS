/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host API/Modules Generator Implementation: parsing utilities
*/

#ifndef JSON_UTILS_H
#define JSON_UTILS_H
#include <vector>
#include <iostream>
#include <string>
#include <set>
#include <rapidjson/document.h>
/**
 * @brief The JSONUtils class contains a set of utilities for JSON parsing
 * They are implemented as static members
 *
 */
class JSONUtils{

public:
    static bool checkField(rapidjson::Value &value, std::string name)
    {
        return value.HasMember(name.c_str());
    }

    static bool checkType(rapidjson::Value &value, std::string name, std::string type)
    {
        const std::string k_type_names_[]=  { "Null", "False", "True", "Object", "Array", "String", "Number"};
        if(type == "string")
            return (k_type_names_[value[name.c_str()].GetType()]=="String");

        if(type == "bool")
            return(k_type_names_[value[name.c_str()].GetType()]=="True" || k_type_names_[value[name.c_str()].GetType()]=="False");

        if(type == "number")
            return (k_type_names_[value[name.c_str()].GetType()]=="Number");

        if(type=="array")
            return (k_type_names_[value[name.c_str()].GetType()]=="Array");

        if(type=="object")
            return (k_type_names_[value[name.c_str()].GetType()]=="Object");

        //default
        return false;
    }

    /**
      Check if a given parsed value is in a set of valid_values
    */
    static bool checkValidValue(const std::string value, const std::set<std::string> valid_values)
    {
        return valid_values.find(value) != valid_values.end();
    }

    /**
     * @brief checkFieldAndType check for the presence of a given field with a given type
     * @param value
     * @param name
     * @param type
     * @return true if everything is ok
     */
    static bool checkFieldAndType(rapidjson::Value &value, std::string field_name, std::string type)
    {
        if(!checkField(value,field_name))
        {
            // std::cerr << "Invalid routine specification: missing property \""<<field_name<<"\""<<std::endl;
            return false;
        }
        if(!checkType(value,field_name,type))
        {
            std::cerr << "Invalid routine specification: property \""<<field_name<<"\" must be of type "<<type<<std::endl;
            return false;
        }
        return true;
    }


};


#endif // JSON_UTILS_H
