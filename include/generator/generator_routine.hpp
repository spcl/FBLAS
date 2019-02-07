/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Host API/Module generator Implementation: routine representation
*/

#ifndef ROUTINE_HPP
#define ROUTINE_HPP
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include "../commons.hpp"

/**
 * @brief The Routine class represents a generic Routine descriptor, containing all the characteristic
 *      of the Routine (width, name, tile sizes...) as indicate by the user in its own JSON file
 *      This is used both for Host Generator and Modules Generators
 *      Some of the fields may be empty
 *
 */
class GeneratorRoutine{
public:
    explicit GeneratorRoutine(std::string blas_name, std::string user_name,std::string precision, FblasArchitecture architecture, unsigned int tier) :
        blas_name_(blas_name),user_name_(user_name), arch_(architecture), tier_(tier)
    {
        if(precision=="single")
            double_precision_=false;
        else
            double_precision_=true;
    }

    /*
     * Setter methods
     */
    void setTransposeA(FblasTranspose tr)
    {
        trans_a_=tr;
    }

    void setTransposeB(FblasTranspose tr)
    {
        trans_b_=tr;
    }

    void setUplo(FblasUpLo ul)
    {
        uplo_=ul;
    }

    void setSide(FblasSide side)
    {
        side_=side;
    }

    void setOrder(FblasOrder ord)
    {
        order_=ord;
    }

    void setWidth(unsigned int par)
    {
        width_=par;
    }

    //Set a 2D computational tile for this routine
    void set2DComputationalTile()
    {
        has_2D_computational_tile_=true;
    }

    void setWidthX(unsigned int width_x)
    {
        width_x_=width_x;
    }

    void setWidthY(unsigned int width_y)
    {
        width_y_=width_y;
    }


    //for level 3
    void setTileSize(unsigned int t)
    {
        tile_size_=t;
    }

    //For level 2
    void setTileNsize(unsigned int t)
    {
        tile_n_size_=t;
    }

    //for level 3
    void setTileMsize(unsigned int t)
    {
        tile_m_size_=t;
    }

    void setIncx(int incx)
    {
        incx_=incx;
    }

    void setIncy(int incy)
    {
        incy_=incy;
    }

    void setLda(unsigned int lda)
    {
        lda_=lda;
    }

    void setLdb(unsigned int ldb)
    {
        ldb_=ldb;
    }

    void setArchitecture(FblasArchitecture arch)
    {
        arch_=arch;
    }

    /*
     * Insert the channel. Return false if already defined
     */
    bool addChannel(std::string routine_ch_name, std::string user_ch_name)
    {
       auto it=channels_.find(routine_ch_name);
       if(it!=channels_.end())
           return false;
       else
       {
           channels_.insert(std::pair<std::string,std::string>(routine_ch_name,user_ch_name));
           return true;
       }
    }

    void setIsKernel(bool value)
    {
        is_kernel_=value;
    }


    void setTilesARowStreamed(bool value)
    {
        tiles_A_row_streamed_=value;
    }

    void setElementsARowStreamed(bool value)
    {
        elements_A_row_streamed_=value;
    }

    void setTilesBRowStreamed(bool value)
    {
        tiles_B_row_streamed_=value;
    }

    void setElementsBRowStreamed(bool value)
    {
        elements_B_row_streamed_=value;
    }

    /*
     * Getter methods
     */

    std::string getBlasName() const
    {
        return blas_name_;
    }

    std::string getUserName() const
    {
        return user_name_;
    }

    bool isDoublePrecision() const
    {
        return double_precision_;
    }

    unsigned int getWidth() const
    {
        return width_;
    }

    bool has2DComputationalTile() const
    {
        return has_2D_computational_tile_;
    }

    unsigned int getWidthX() const
    {
        return width_x_;
    }
    unsigned int getWidthY() const
    {
        return width_y_;
    }

    //returns if the matrix A is transposed or not
    FblasTranspose getTransposeA() const
    {
        return trans_a_;
    }

    FblasTranspose getTransposeB() const
    {
        return trans_b_;
    }

    FblasUpLo getUplo() const
    {
        return uplo_;
    }


    FblasOrder getOrder() const
    {
        return order_;
    }

    FblasSide getSide() const
    {
        return side_;
    }

    unsigned int getTileNsize() const
    {
        return tile_n_size_;
    }

    unsigned int getTileMsize() const
    {
        return tile_m_size_;
    }

    unsigned int getTileSize() const
    {
        return tile_size_;
    }


    int getIncx() const
    {
        return incx_;
    }

    int getIncy() const
    {
        return incy_;
    }

    unsigned int getLda() const
    {
        return lda_;
    }

    unsigned int getLdb() const
    {
        return ldb_;
    }

    FBlasArchiteture getArchitecture() const
    {
        return arch_;
    }

    /**
     * @brief getChannelName returns the channel name defined by the user
     * @param routine_channel_name the channel that we are looking for
     * @return
     */
    std::string getChannelName(std::string routine_channel_name) const
    {
        return channels_.at(routine_channel_name);
    }

    bool isKernel() const
    {
        return is_kernel_;
    }

    bool areTilesARowStreamed() const
    {
        return tiles_A_row_streamed_;
    }

    bool areElementsARowStreamed() const
    {
        return elements_A_row_streamed_;
    }

    bool areTilesBRowStreamed() const
    {
        return tiles_B_row_streamed_;
    }

    bool areElementsBRowStreamed() const
    {
        return elements_B_row_streamed_;
    }


    void print()
    {
        std::cout << "Routine : " << blas_name_<<std::endl;
        std::cout << "User name: " <<user_name_ <<std::endl;
        if(double_precision_)
            std::cout << "Precision: double" <<std::endl;
        else
            std::cout << "Precision: single" <<std::endl;

        std::cout << "Required fields: ";
        if(trans_a_!=FBLAS_T_UNDEF)
            if(trans_a_==FBLAS_NO_TRANSPOSED)
                std::cout << "Non Transposed ";
            else
                std::cout << "Transposed ";
        if(uplo_ != FBLAS_UL_UNDEF)
            if(uplo_==FBLAS_LOWER)
                std::cout << "Lower Triangular ";
            else
                std::cout << "Upper Triangular ";

        if(order_!=FBLAS_O_UNDEF)
            if(order_==FBLAS_ROW_MAJOR)
                std::cout << "Row Major ";
            else
                std::cout << "Column Major ";
        std::cout << std::endl;
        std::cout << "Optional fields: "<<std::endl;
        if(width_!=0)
            std::cout << " Width_ = " << width_;
        if(tier_==2)
        {
            std::cout << "incx: " << incx_ << std::endl;
            std::cout << "incy: " << incy_ << std::endl;
        }

        if(tier_==1)
        {
            std::cout << "Channels: "<<std::endl;
            for(auto ch:channels_)
                std::cout << ch.first << ": " <<ch.second<<std::endl;
        }


    }


private:
    //name of the routine according to blas (without indication of the precision)
    std::string blas_name_;

    //user name for the routine
    std::string user_name_;

    //degree of spatial parallelism (optional)
    unsigned int width_=0;

    //single or double precision
    bool double_precision_;

    /*
     * Fields that may be not required by all the routines
     */
    FblasOrder order_=FBLAS_O_UNDEF;
    FblasTranspose trans_a_=FBLAS_T_UNDEF;
    FblasTranspose trans_b_=FBLAS_T_UNDEF;
    FblasUpLo uplo_=FBLAS_UL_UNDEF;
    FblasSide side_=FBLAS_SIDE_UNDEF;
    FblasArchitecture arch_=FBLAS_STRATIX_10;
    //tiling
    unsigned int tile_n_size_=0;
    unsigned int tile_m_size_=0;

    //strides
    int incx_=1;
    int incy_=1;

    unsigned int lda_=0; //the default value here is 0, meaning that the matrix will be accessed with a classical stride
    unsigned int ldb_=0; //the default value here is 0, meaning that the matrix will be accessed with a classical stride

    bool has_2D_computational_tile_=false;  //true if the routine is implemented using a 2D computation tile (e.g. GEMM, SYRK)
    unsigned int width_x_=0;                //in case it uses a 2D computational tile this one express the number of columns
    unsigned int width_y_=0;                //and this one the number of rows
    unsigned int tile_size_=0;              //the memory tile size


    //used by the module generator (tier 1)
    std::map<std::string,std::string> channels_; //set of channels (name required -> name indicated by the user
    bool is_kernel_; //true if it must be generated as kernel, false if the user wants a function
    //used for level-2, level-3, indicates wheter the order of the tiles (for A and B) and of the elements
    bool tiles_A_row_streamed_=true;
    bool elements_A_row_streamed_=true;
    bool tiles_B_row_streamed_=true;
    bool elements_B_row_streamed_=true;

    unsigned int tier_;

};

#endif // ROUTINE_HPP
