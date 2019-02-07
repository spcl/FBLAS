/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host API Implementation: host code generator commons
*/

#ifndef GENERATOR_COMMONS_HPP
#define GENERATOR_COMMONS_HPP


#if !defined(BASE_FOLDER)
//#define BASE_FOLDER "/home/dematteis/remote_fs/work/fpga/src/fpga_blas/" //skeleton folder
#define BASE_FOLDER "../fpga_blas/" //skeleton folder
#endif
/**
  Constant used in parsing and code generation
*/
static const unsigned int k_default_width_=32;          //default value for the width
static const unsigned int k_default_tiling_=256;       //default value for the tiling
static const unsigned int k_default_2d_width_=8;        //default width (x or y) in case of 2D computational tiling




//Colored print
#define RED_UNDERLINE "\x1b[31;4m"
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define GREEN_BOLD   "\x1b[32;1m"
#define RED_BOLD "\x1b[31;1m"
#define YELLOW  "\x1b[33m"
#define BLUE    "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN    "\x1b[36m"
#define CYAN_BOLD "\x1b[36;1m"
#define RESET   "\x1b[0m"
#endif // GENERATOR_COMMONS_HPP
