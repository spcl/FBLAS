CXX	    = g++
AOCXX   = aoc
CXXFLAGS= -std=c++11 -g
#This should get automatically the first device. Change if you need another board
BOARD 	= $(shell $(AOCXX) -list-boards | sed -n '2p' | tr -d " ")
AOCXXFLAGS  = -march=emulator -board=$(BOARD) 
# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
GEN = ../bin/host_generator
RAPIDJSON_DIR = ../rapidjson/include 

.PHONY: host_generator

all: report 

host_generator: 
	@if [ ! -f $(GEN) ] ; \
	then \
	    make -C ../ host_generator; \
	fi;


emulated: generator tutorial.json
	$(GEN) tutorial.json /tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/sdot.cl /tmp/sscal.cl -I../blas/include

tutorial: tutorial.cpp
	$(CXX) tutorial.cpp $(CXXFLAGS) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -o tutorial -I$(RAPIDJSON_DIR) -I../include/

run_tutorial: tutorial emulated
	@echo "Executing with n=1024 and alpha=2..."
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./tutorial -b ./sscal.aocx -j /tmp/generated_routines.json -n 1024 -a 2

clean:
	rm -fr sscal/ || true
	rm -rf sscal*.ao* || true
	rm -rf sdot*.ao* || true
	rm tutorial || true


