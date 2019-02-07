/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Modules Generator Implementation: helper representation
*/

#ifndef GENERATOR_HELPER_HPP
#define GENERATOR_HELPER_HPP
#include <string>

/**
 * @brief The GeneratorHelper class represent an Helper. It is used by the Modules Generator
 */
class GeneratorHelper{

public:
    explicit GeneratorHelper(std::string type, std::string user_name,std::string precision, std::string channel_name):
        type_(type), channel_name_(channel_name), user_name_(user_name)
    {
        if(precision=="single")
            double_precision_=false;
        else
            double_precision_=true;
    }

    void setStride(int stride)
    {
        stride_=stride;
    }

    void setWidth(unsigned int width)
    {
        width_=width;
    }

    std::string getType() const
    {
        return type_;
    }

    std::string getUserName() const
    {
        return user_name_;
    }

    std::string getChannelName() const
    {
        return channel_name_;
    }
    void setTilesRowStreamed(bool value)
    {
        tiles_row_streamed_=value;
    }

    void setElementsRowStreamed(bool value)
    {
        elements_row_streamed_=value;
    }


    void setTileNsize(unsigned int t)
    {
        tile_n_size_=t;
    }

    //for level 3
    void setTileMsize(unsigned int t)
    {
        tile_m_size_=t;
    }


    //------------------------------------------------
    //Getter
    //--------------------------

    bool isDoublePrecision() const
    {
        return double_precision_;
    }
    int getStride() const
    {
        return stride_;
    }

    bool areTilesRowStreamed() const
    {
        return tiles_row_streamed_;
    }

    bool areElementsRowStreamed() const
    {
        return elements_row_streamed_;
    }

    unsigned int getWidth() const
    {
        return width_;
    }
    unsigned int getTileNsize() const
    {
        return tile_n_size_;
    }

    unsigned int getTileMsize() const
    {
        return tile_m_size_;
    }


private:
    std::string type_;
    std::string user_name_;
    std::string channel_name_;
    bool double_precision_;
    int stride_=1; //if present
    unsigned int width_=16;
    //tiling
    unsigned int tile_n_size_=0;
    unsigned int tile_m_size_=0;
    //in case it refers to a Matrix
    bool tiles_row_streamed_=true;
    bool elements_row_streamed_=true;



};
#endif // GENERATOR_HELPER_HPP
