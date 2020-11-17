CXX     = g++
AOCXX   = aoc
CXXFLAGS= -std=c++11 -O3 -pthread -lgtest
#This should get automatically the first device. Change if you need another board
BOARD 	= $(shell $(AOCXX) -list-boards | sed -n '2p' | tr -d " ")
AOCXXFLAGS  = -march=emulator -board=$(BOARD) 
AOCL_COMPILE_CONFIG := $(shell aocl compile-config) 
AOCL_LINK_CONFIG := $(shell aocl link-config)
RAPIDJSON_DIR = ../../rapidjson/include 
GEN = ../../codegen/host_codegen.py
BIN_DIR=bin
LEVEL_1 = test_iamax test_dot test_copy test_rot test_rotg test_swap
LEVEL_2 = test_gemv test_ger test_syr
LEVEL_3 = test_gemm test_syrk test_syr2k test_trsm
.PHONY = clean host_generator 

	
test_%: $(BIN_DIR)
	@echo "Executing test_$*"
	@echo "Compiling for emulation ..."
	rm -f /tmp/test_*$**cl || true
	rm -f test_*"$*"*aoc* || true
	$(GEN) $*.json -output_dir=/tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/test_*$**cl -I../../blas/include
	$(CXX) $(CXXFLAGS) test_$*.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) -I$(RAPIDJSON_DIR) -o $(BIN_DIR)/$@ -lgtest
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 $(BIN_DIR)/$@ $$(ls test_*$**aocx) /tmp/generated_routines.json

test_iamax: $(BIN_DIR)
	@echo "Executing test_iamax"
	@echo "Compiling for emulation ..."
	rm -f /tmp/test_*i*amax*cl || true
	rm -f test_*i*amax*aocx || true
	$(GEN) iamax.json -output_dir=/tmp/
	$(AOCXX) $(AOCXXFLAGS) /tmp/test_i*max*cl -I../../blas/include
	$(CXX) $(CXXFLAGS) test_iamax.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) -I$(RAPIDJSON_DIR) -o $(BIN_DIR)/$@ -lgtest
	env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 $(BIN_DIR)/$@ $$(ls test_i*max*aocx) /tmp/generated_routines.json

test_level1: $(BIN_DIR)
	make ${LEVEL_1}
	
test_level2: $(BIN_DIR)
	make ${LEVEL_2}

test_level3: $(BIN_DIR)
	make ${LEVEL_3}

$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r test_*/ || true
	rm -fr $(BIN_DIR) || true

