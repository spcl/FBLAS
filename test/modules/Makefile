CXX     = g++
AOCXX   = aoc
CXXFLAGS= -std=c++11 -O3 -pthread -lgtest
#This should get automatically the first device. Change if you need another board
BOARD 	= $(shell $(AOCXX) -list-boards | sed -n '2p' | tr -d " ")
AOCXXFLAGS  = -march=emulator -board=$(BOARD)
AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)
RAPIDJSON_DIR = /home/tdematt/lib/rapidjson/include
GEN = ../../bin/modules_generator
BIN_DIR=bin

.PHONY = clean generator

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
#TODO fix this for code release
#GTEST_DIR = ..
#CPPFLAGS += -isystem $(GTEST_DIR)/include

generator:
		@if [ ! -f $(GEN) ] ; \
		then \
			make -C ../../ modules_generator; \
		fi;

test_%: generator $(BIN_DIR)
		@echo "Executing test_$*"
		@echo "Compiling for emulation ..."
		rm -f /tmp/test_*$**cl || true
		rm -f test_*"$*"*aocx || true
		$(GEN) $*.json /tmp/
		$(AOCXX) $(AOCXXFLAGS) /tmp/test_*$**cl -I../../blas/include
		$(CXX) $(CXXFLAGS) test_$*.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) -I$(RAPIDJSON_DIR) -o $(BIN_DIR)/$@
		$(eval binfile:=$(shell find . -iname test_*"$*"*aocx))
		@echo "Executing tests..." $(binfile)
		env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 $(BIN_DIR)/$@ $(binfile)
		$(unset binfile)



$(BIN_DIR):
		mkdir $(BIN_DIR)

clean:
		rm -f *.aoc* || true
		rm -fr *.temp || true
		rm -r test_*/ || true
		rm -fr $(BIN_DIR) || true

