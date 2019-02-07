CXX	= g++
CXXFLAGS= -std=c++11 
INC=rapidjson/include/
TARGET_DIR=bin
HOST_GEN_SRC_DIR := src/generator
HOST_GEN_SRC := $(HOST_GEN_SRC_DIR)/host_api_level1_generators.cpp $(HOST_GEN_SRC_DIR)/host_api_level2_generators.cpp $(HOST_GEN_SRC_DIR)/host_api_level3_generators.cpp $(HOST_GEN_SRC_DIR)/host_api_generator_defs.cpp  $(HOST_GEN_SRC_DIR)/host_generator.cpp 
MODULE_GEN_SRC := $(HOST_GEN_SRC_DIR)/modules_generator.cpp $(HOST_GEN_SRC_DIR)/module_generator_defs.cpp $(HOST_GEN_SRC_DIR)/modules_level1_generators.cpp $(HOST_GEN_SRC_DIR)/modules_level2_generators.cpp $(HOST_GEN_SRC_DIR)/modules_helper_generators.cpp

.PHONY : all

## GENERATORS

host_generator: $(HOST_GEN_SRC) $(TARGET_DIR)
		$(CXX) -o bin/host_generator $(HOST_GEN_SRC)  -I$(INC) -DBASE_FOLDER="\"$(shell pwd)\""

modules_generator: $(MODULE_GEN_SRC) $(TARGET_DIR)
		$(CXX) -o bin/modules_generator $(MODULE_GEN_SRC)  -I$(INC) -DBASE_FOLDER="\"$(shell pwd)\""

all : host_generator modules_generator

$(TARGET_DIR):
		mkdir $(TARGET_DIR)