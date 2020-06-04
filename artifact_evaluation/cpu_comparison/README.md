This folder contains the different program used for comparing FPGA designs with CPU programs.


Being `<test_name>` in `{sdot, ddot, sgemv, dgemv}`, the following 
commands can be used:
 - compile emulation `make <test_name>_emulator`
 - compile host program `make dot/gemv_host`
 - synthesize hardware `make <test_name>_hardware`
 
 
To execute a program:
- if in emulation, set the environment variable ` CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1`
- execute the host program locate under `bin/` by passing the suggested parameters. 
 
 
Note: for the evaluation of `GEMM`, please use the corresponding program in the `sample/invidual_modules` folder.

