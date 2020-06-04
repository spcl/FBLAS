In this folder, there are the files used for the evaluation of single FBLAS modules
in isolation.

To capture different computational and communication complexities, for the moment we consider
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
- if in emulation, set the environment variable ` CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1`
- execute the host program locate under `bin/` by passing the suggested parameters. Note: for GEMM
    it will require the json file automatically generated and located under `<s/d>gemm_codegen_files/generated_routines.json`
 
