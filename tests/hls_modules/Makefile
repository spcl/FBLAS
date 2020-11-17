CXX     = g++
AOCXX   = aoc
CXXFLAGS= -std=c++11 -O3 -pthread -lgtest
#This automatically gets the first device. Change if you need another board
BOARD 	= $(shell $(AOCXX) -list-boards | sed -n '2p' | tr -d " ")
AOCXXFLAGS  = -march=emulator -board=$(BOARD)
AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)
RAPIDJSON_DIR = ../../rapidjson/include 
GEN = ../../codegen/modules_codegen.py
BIN_DIR=bin
.PHONY = clean generator


test_%: $(BIN_DIR)
	@echo "Executing test_$*"
	@echo "Compiling for emulation ..."
	rm -f /tmp/test_*$**cl || true
	rm -f test_*"$*"*aocx || true
	$(GEN) $*.json -output_dir=/tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/test_*$**cl -I../../blas/include
	$(CXX) $(CXXFLAGS) test_$*.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) -I$(RAPIDJSON_DIR) -o $(BIN_DIR)/$@ -lgtest
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 $(BIN_DIR)/$@ $$(ls test_*"$*"*aocx)


$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r test_*/ || true
	rm -fr $(BIN_DIR) || true

