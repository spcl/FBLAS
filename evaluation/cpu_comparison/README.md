This folder contains the different programs used for comparing FPGA designs with CPU programs.


Being `<test_name>` in `{sdot, ddot, sgemv, dgemv}`, the following 
commands can be used:
 - compile emulation `make <test_name>_emulator`
 - compile host program `make dot/gemv_host`
 - synthesize hardware `make <test_name>_hardware`
 
 
To execute a program:
- if executing in emulation, set the environment variable ` CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1`
- execute the host program locate under `bin/` by passing the suggested command line parameters. 
 
 
Every host program accepts, among the other, the number of runs to execute and produces the averaged execution time,
as well as confidence intervals.
The result is validated against openblas (CPU) version of the same computation.

Note: for the evaluation of `GEMM`, please use the corresponding program in the `sample/invidual_modules` folder.

