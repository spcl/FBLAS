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
GEN = ../../../codegen/modules_codegen.py
BIN_DIR=bin
.PHONY = clean generator $(BIN_DIR)

define codegen_files
	@echo "Codegen files..."
	rm -f $(1)_codegen_files/*
	mkdir -p $(1)_codegen_files
	$(GEN) $(1).json -output_dir=$(1)_codegen_files
endef


################################### BICG #################################
sbicg_emulator:
	@echo "Compiling sbicg for emulation ..."
	$(call codegen_files,sbicg)
	#removing previous bitstream
	rm sbicg.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator bicg_helpers.cl sbicg_codegen_files/*bicg*cl
	mv  sbicg_write_s.aocx sbicg.aocx
	@echo "Generated file sbicg.aocx"

sbicg_hardware:
	@echo "Compiling sbicg for emulation ..."
	$(call codegen_files,sbicg)
	#removing previous bitstream
	rm sbicg.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -report bicg_helpers.cl sbicg_codegen_files/*bicg*cl
	mv sbicg_write_s.aocx sbicg_hardware.aocx
	mv sbicg_write_s sbicg_hardware
	@echo "Generated file sbicg_hardware.aocx"

dbicg_emulator:
	@echo "Compiling dbicg for emulation ..."
	$(call codegen_files,dbicg)
	#removing previous bitstream
	rm dbicg.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator bicg_helpers.cl dbicg_codegen_files/*bicg*cl -DDOUBLE
	mv  dbicg_write_s.aocx dbicg.aocx
	@echo "Generated file dbicg.aocx"

dbicg_hardware:
	@echo "Compiling dbicg for emulation ..."
	$(call codegen_files,dbicg)
	#removing previous bitstream
	rm dbicg.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -report bicg_helpers.cl dbicg_codegen_files/*bicg*cl -DDOUBLE
	mv dbicg_write_s.aocx dbicg_hardware.aocx
	mv dbicg_write_s dbicg_hardware
	@echo "Generated file dbicg_hardware.aocx"




bicg_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) streaming_bicg.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@

################################ GEMVER ###########################################

sgemver_emulator:
	@echo "Compiling sgemver for emulation ..."
	$(call codegen_files,sgemver)
	#removing previous bitstream
	rm sgemver.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator gemver_helpers.cl sgemver_codegen_files/*gemver*cl
	mv  sgemver_write_x.aocx sgemver.aocx
	@echo "Generated file sgemver.aocx"

sgemver_hardware:
	@echo "Compiling sgemver for hardware ..."
	$(call codegen_files,sgemver)
	#removing previous bitstream
	rm sgemver.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -report gemver_helpers.cl sgemver_codegen_files/*gemver*cl
	mv sgemver_write_x.aocx sgemver_hardware.aocx
	mv sgemver_write_x sgemver_hardware
	@echo "Generated file sgemver_hardware.aocx"

dgemver_emulator:
	@echo "Compiling dgemver for emulation ..."
	$(call codegen_files,dgemver)
	#removing previous bitstream
	rm dgemver.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator gemver_helpers.cl dgemver_codegen_files/*gemver*cl -DDOUBLE
	mv  dgemver_write_x.aocx dgemver.aocx
	@echo "Generated file dgemver.aocx"

dgemver_hardware:
	@echo "Compiling dgemver for hardware ..."
	$(call codegen_files,dgemver)
	#removing previous bitstream
	rm dgemver.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -report gemver_helpers.cl dgemver_codegen_files/*gemver*cl -DDOUBLE
	mv dgemver_write_x.aocx dgemver_hardware.aocx
	mv dgemver_write_x dgemver_hardware
	@echo "Generated file dgemver_hardware.aocx"


gemver_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) streaming_gemver.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@



################################ AXPYDOT ###########################################

%_emulator:
	@echo "Compiling for emulation ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS) -march=emulator $*_codegen_files/*$**cl
	mv $*_write_beta.aocx $*.aocx
	@echo "Generated file $*.aocx"

%_hardware:
	@echo "Compiling for hardware ..."
	rm -f $*_codegen_files/*
	mkdir -p $*_codegen_files
	$(GEN) $*.json -output_dir=$*_codegen_files
	$(AOCXX) $(AOCXXFLAGS) $*_codegen_files/*$**cl -report
	mv $*_write_beta.aocx $*_hardware.aocx
	mv $*_write_beta $*_hardware
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
	$(CXX) $(CXXFLAGS) streaming_axpydot.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@



$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r *_codegen_files || true
	rm -fr $(BIN_DIR) || true

