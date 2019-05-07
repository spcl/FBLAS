CXX	    = g++
AOCXX   = aoc
CXXFLAGS= -std=c++11 -g -lpthread
#This should get automatically the first device. Change if you need another board
BOARD 	= $(shell $(AOCXX) -list-boards | sed -n '2p' | tr -d " ")
AOCXXFLAGS  = -march=emulator -board=$(BOARD) 
# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )
GEN = ../bin/host_generator
MODULES_GEN = ../bin/modules_generator
RAPIDJSON_DIR = ../rapidjson/include 

.PHONY: host_generator modules_generator


host_generator: 
	@if [ ! -f $(GEN) ] ; \
	then \
	    make -C ../ host_generator; \
	fi;

modules_generator:
	@if [ ! -f $(MODULES_GEN) ] ; \
	then \
	    make -C ../ modules_generator; \
	fi;


#Tutorial

emulated: host_generator tutorial.json
	$(GEN) tutorial.json /tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/sdot.cl /tmp/sscal.cl -I../blas/include

tutorial: tutorial.cpp
	$(CXX) tutorial.cpp $(CXXFLAGS) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -o tutorial -I$(RAPIDJSON_DIR) -I../include/

run_tutorial: tutorial emulated
	@echo "Executing with n=1024 and alpha=2..."
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./tutorial -b ./sscal.aocx -j /tmp/generated_routines.json -n 1024 -a 2

#Axpydot

fblas_axpydot_emulated: host_generator fblas_axpydot.json
	$(GEN) fblas_axpydot.json /tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/axpydot_fblas*.cl -I../blas/include 

streamed_axpydot_emulated: modules_generator streamed_axpydot.json
	$(MODULES_GEN) streamed_axpydot.json /tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/axpydot_streamed*.cl -I../blas/include 

fblas_axpydot_host: fblas_axpydot.cpp
	$(CXX) fblas_axpydot.cpp $(CXXFLAGS) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -o fblas_axpydot -I$(RAPIDJSON_DIR) -I../include/ -lopenblas

streamed_axpydot_host: streamed_axpydot.cpp
	$(CXX) streamed_axpydot.cpp $(CXXFLAGS) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) -o streamed_axpydot -I$(RAPIDJSON_DIR) -I../include/ -lopenblas

run_axpydot: fblas_axpydot_emulated fblas_axpydot_host streamed_axpydot_emulated streamed_axpydot_host
	@echo "*********************************************************"
	@echo "Executing FBLAS version with n=1024 and alpha=2, 10 runs"
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./fblas_axpydot -b ./axpydot_fblas_sdot.aocx -j /tmp/generated_routines.json -n 1024 -a 2 -r 10
	@echo "*********************************************************"
	@echo "Executing Streamed version with n=1024 and alpha=2, 10 runs"
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./streamed_axpydot -b ./axpydot_streamed_write_scalar.aocx  -n 1024 -a 2 -r 10

clean:
	rm -fr sscal/ || true
	rm -rf sscal*.ao* || true
	rm -rf sdot*.ao* || true
	rm -rf axpydot_fblas_* || true
	rm -rf axpydost_streamed_* || true
	rm tutorial || true


