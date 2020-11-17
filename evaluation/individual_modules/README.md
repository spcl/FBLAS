In this folder, there are the programs used for the evaluation of single FBLAS modules
in isolation (sec. 6.B).

To capture different computational and communication complexities, we consider
modules that implement the `DOT`, `GEMV` and `GEMM` routines, as representative samples of BLAS Level 1, 2, and 3, respectively.

For `DOT` and `GEMV`, input data is generated directly on the FPGA, to test the scaling behavior of these memory bound applications 
, considering vectorization width that can exploit memory interfaces faster than the one offered by the testbed (e.g., HBM).

For each test case, there is:
- the json file with the description of the involved module
- an host file for executing the program

Being `<test_name>` in `{sdot, ddot, sgemv, dgemv, sgemm, dgemm}`, the following 
commands can be used:
 - compile emulation `make <test_name>_emulator`
 - compile host program `make dot/gemv/gemm_host`
 - compile report `makte <test_name>_report`
 
To execute a program:
- for emulating a design, set the environment variable ` CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1` as suggested by Intel
- execute the host program locate under `bin/` by passing the suggested parameters (execute the program without argument to obtain a list). Note: for GEMM it will require the json file automatically generated and located under `<s/d>gemm_codegen_files/generated_routines.json`
 
Every host program accepts, among the other, the number of runs to execute and produces the averaged execution time,
as well as confidence intervals.
The result is validated against openblas (CPU) version of the same computation.


For example, for compiling and executing matrix-vector product in single precision and emulation:

```Bash
# Create emulation bitstream (generates sdot.aocx)
$ make sgemv_emulator
# Compile host program
$ make gemv_host
#execute for emulation, 5 runs. The program takes in input sizes (n,m), alpha and beta multipliers (a,b) and tile sizes (k,j) as defined in the json description
$ env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1  bin/gemv_host -b sgemv.aocx -n 2048 -m 1024 -a 2 -c 1 -k 1024 -j 1024 -r 1 -p single  

```

Notes: 
- for GEMM, single precision, the emulation could sometimes fails, however in hardware it works correctly. This depends on the used compiler and tile size: please try to reduce it. The same problem does not happens in double precision.
- for GEMM, changing the compilation seed for synthesis(by adding the `-seed=<number>` at the end of compilation command) could result in higher/lower synthesis frequency.