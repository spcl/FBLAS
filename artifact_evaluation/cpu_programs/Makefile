CXX     = g++
AOCXX   = aoc
CXXFLAGS= -std=c++11 -O3 -pthread
#MKL FLAGS: change accordingly
MKL_FLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -DMKL_ILP64 -m64 -I${MKLROOT}/include
# Mammut (power measurement) flags: change accordingly
MAMMUT_ROOT = ${HOME}
MAMMUT_FLAGS = -I${MAMMUT_ROOT}/mammut/include/ -L${MAMMUT_ROOT}/mammut/lib/ -lmammut

BIN_DIR=bin
.PHONY = clean generator $(BIN_DIR)

%_program: $(BIN_DIR)
	@echo "Compiling $* ..."
	$(CXX) $(CXXFLAGS) $*.cpp  -o $(BIN_DIR)/$@ $(MKL_FLAGS)  $(MAMMUT_FLAGS) -DPOWER_MEASUREMENT

$(BIN_DIR):
	mkdir $(BIN_DIR)