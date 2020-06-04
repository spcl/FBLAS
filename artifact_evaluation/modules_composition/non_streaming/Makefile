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
GEN = ../../../codegen/host_codegen.py
BIN_DIR=bin
RAPIDJSON = ../../../rapidjson/include/
.PHONY = clean generator $(BIN_DIR)

define codegen_files
	@echo "Codegen files..."
	rm -f $(1)_codegen_files/*
	mkdir -p $(1)_codegen_files
	$(GEN) $(1).json -output_dir=$(1)_codegen_files
endef


################################ AXPYDOT ###########################################

%_emulator:
	@echo "Compiling for emulation ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS) -march=emulator $*_codegen_files/*$**cl
	mv $**.aocx $*.aocx
	@echo "Generated file $*.aocx"

%_hardware:
	@echo "Compiling for hardware ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS) $*_codegen_files/*$**cl -report
	mv $**.aocx $*_hardware.aocx
	mv $** $*_hardware
	@echo "Generated file $*_hardware.aocx"

%_report:
	@echo "Generating report ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS)  -rtl $*_codegen_files/*$**cl
	@echo "Report generated"





axpydot_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) axpydot_host.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@ -I$(RAPIDJSON)

bicg_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) bicg_host.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@ -I$(RAPIDJSON)

gemver_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) gemver_host.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@ -I$(RAPIDJSON)

$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r *_codegen_files || true
	rm -fr $(BIN_DIR) || true

