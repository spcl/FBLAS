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
BIN_DIR=bin
.PHONY = clean generator $(BIN_DIR)


define codegen_files
	@echo "Codegen files..."
	rm -f $(1)_codegen_files/*
	mkdir -p $(1)_codegen_files
	$(GEN) $(1).json -output_dir=$(1)_codegen_files
endef

########################### DOT ######################################
sdot_emulator:
	@echo "Compiling sdot for emulation ..."
	$(call codegen_files,sdot)
	#removing previous bitstream
	rm sdot.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator sdot_codegen_files/*sdot*cl dot_helpers.cl
	mv dot_helpers.aocx sdot.aocx
	@echo "Generated file sdot.aocx"


sdot_hardware:
	@echo "Compiling sdot for hardware ..."
	$(call codegen_files,sdot)
	$(AOCXX) $(AOCXXFLAGS) dot_helpers.cl sdot_codegen_files/*sdot*cl -report
	mv *sdot*.aocx sdot_hardware.aocx
	#mv dot_helpers sdot_hardware
	@echo "Generated file sdot_hardware.aocx"

ddot_emulator:
	@echo "Compiling ddot for emulation ..."
	$(call codegen_files,ddot)
	#removing previous bitstream
	rm ddot.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator ddot_codegen_files/*ddot*cl dot_helpers.cl -DDOUBLE
	mv dot_helpers.aocx ddot.aocx
	@echo "Generated file ddot.aocx"


ddot_hardware:
	@echo "Compiling ddot for hardware ..."
	$(call codegen_files,ddot)
	$(AOCXX) $(AOCXXFLAGS) ddot_codegen_files/*ddot*cl dot_helpers.cl -DDOUBLE -report
	mv dot_helpers.aocx ddot_hardware.aocx
	mv dot_helpers ddot_hardware
	@echo "Generated file ddot_hardware.aocx"


dot_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_dot.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@

########################### GEMV ######################################

%_emulator:
	@echo "Compiling $* for emulation ..."
	$(call codegen_files,$*)
	#removing previous bitstream
	rm $*.aocx || true
	$(AOCXX) $(AOCXXFLAGS) -march=emulator $*_codegen_files/*$**cl
	mv  $*_write_vector.aocx $*.aocx
	@echo "Generated file $*.aocx"

%_hardware:
	@echo "Compiling $* for hardware ..."
	$(call codegen_files,$*)
	$(AOCXX) $(AOCXXFLAGS) $*_codegen_files/*$**cl  -report
	mv $*_write_vector.aocx $*_hardware.aocx
	mv $*_write_vector $*_hardware
	@echo "Generated file $*_hardware.aocx"


gemv_host: $(BIN_DIR)
	@echo "Compiling host ..."
	$(CXX) $(CXXFLAGS) host_gemv.cpp $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(BLASFLAGS) -o $(BIN_DIR)/$@

$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm -f *.aoc* || true
	rm -fr *.temp || true
	rm -r *_codegen_files || true
	rm -fr $(BIN_DIR) || true

