CXX     = g++
AOCXX   = aoc
#BLAS compilation flag: change if needed
MKL_FLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -DMKL_ILP64 -m64 -I${MKLROOT}/include
CXXFLAGS= -std=c++11  -pthread
#Change if you need another board
BOARD 	= p520_max_sg280l
AOCXXFLAGS  = -board=$(BOARD) -fp-relaxed -fpc -no-interleaving=default
AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)
GEN = ../../codegen/modules_codegen.py
BIN_DIR=bin
.PHONY = clean generator $(BIN_DIR)


define codegen_files
	@echo "Codegen files..."
	rm -f $(1)_codegen_files/*
	mkdir -p $(1)_codegen_files
	$(GEN) $(1).json -output_dir=$(1)_codegen_files
endef


gemm_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_gemm.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(MKL_FLAGS) -o $(BIN_DIR)/$@


trsm_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_trsm.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(MKL_FLAGS) -o $(BIN_DIR)/$@

$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r *_codegen_files || true
	rm -fr $(BIN_DIR) || true

