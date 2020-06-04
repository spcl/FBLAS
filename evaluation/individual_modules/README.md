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
- execute the host program locate under `bin/` by passing the suggested parameters. Note: for GEMM
    it will require the json file automatically generated and located under `<s/d>gemm_codegen_files/generated_routines.json`
 
Every host program accepts, among the other, the number of runs to execute and produces the averaged execution time,
as well as confidence intervals.
The result is validated against openblas (CPU) version of the same computation.