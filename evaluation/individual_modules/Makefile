CXX     = g++
AOCXX   = aoc
#BLAS compilation flag: change if needed
BLASFLAGS = -lopenblas
CXXFLAGS= -std=c++11 -O3 -pthread
#Change if you need another board
BOARD 	= p520_max_sg280l
AOCXXFLAGS  = -board=$(BOARD) -fp-relaxed -fpc -no-interleaving=default
AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)
GEN = ../../codegen/modules_codegen.py
HOST_GEN = ../../codegen/host_codegen.py
RAPIDJSON = ../../rapidjson/include/
BIN_DIR=bin
.PHONY = clean generator $(BIN_DIR)


# Gemm is generated with host codegen as data is in memory

%gemm_emulator:
	@echo "Compiling for emulation ..."
	rm -f $*gemm_codegen_files/*
	mkdir -p $*gemm_codegen_files
	$(HOST_GEN) $*gemm.json -output_dir=$*gemm_codegen_files
	$(AOCXX) $(AOCXXFLAGS) -march=emulator $*gemm_codegen_files/*$**cl
	mv *$*gemm*.aocx $*gemm.aocx || true
	@echo "Generated file $*gemm.aocx"

%gemm_report:
	@echo "Generating report ..."
	rm -f $*gemm_codegen_files/*
	mkdir -p $*gemm_codegen_files
	$(HOST_GEN) $*gemm.json -output_dir=$*gemm_codegen_files
	$(AOCXX) $(AOCXXFLAGS)  -rtl $*gemm_codegen_files/*$*gemm*cl
	@echo "Report generated"

%gemm_hardware:
	@echo "Compiling for hardware ..."
	rm -f $*gemm_codegen_files/*
	mkdir -p $*gemm_codegen_files
	$(HOST_GEN) $*gemm.json -output_dir=$*gemm_codegen_files
	$(AOCXX) $(AOCXXFLAGS) $*gemm_codegen_files/*$*gemm*cl -report
	mv *$*gemm.aocx $*gemm_hardware.aocx || true
	mv *$*gemm $*gemm_hardware || true
	@echo "Generated file $*gemm_hardware.aocx"

%_emulator:
	@echo "Compiling for emulation ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS) -march=emulator $*_codegen_files/*$**cl
	mv *$**.aocx $*.aocx
	@echo "Generated file $*.aocx"

%_hardware:
	@echo "Compiling for hardware ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS) $*_codegen_files/*$**cl -report
	mv *$**.aocx $*_hardware.aocx
	@echo "Generated file $*_hardware.aocx"

%_report:
	@echo "Generating report ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS)  -rtl $*_codegen_files/*$**cl
	@echo "Report generated"

dot_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_dot.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@

gemv_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_gemv.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@

gemm_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_gemm.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@ -I$(RAPIDJSON)

$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r *_codegen_files || true
	rm -fr $(BIN_DIR) || true

