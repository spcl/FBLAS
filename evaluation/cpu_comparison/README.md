This folder contains the different programs used for comparing FPGA designs with CPU programs.


Being `<test_name>` in `{sdot, ddot, sgemv, dgemv}`, the following 
commands can be used:
 - compile emulation `make <test_name>_emulator`
 - compile host program `make dot/gemv_host`
 - synthesize hardware `make <test_name>_hardware`
 
 
To execute a program:
- if executing in emulation, set the environment variable ` CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1`
- execute the host program locate under `bin/` by passing the suggested command line parameters(launch the program without arguments to obtain a list). 
 
 
Every host program accepts, among the other, the number of runs to execute and produces the averaged execution time,
as well as confidence intervals.
The result is validated against openblas (CPU) version of the same computation.


For example, for compiling and executing dot product in single precision and emulation:

```Bash
# Create emulation bitstream (generates sdot.aocx)
make sdot_emulator
# Compile host program
make dot_host
#execute for emulation, 5 runs
env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1  bin/dot_host -b sdot.aocx -n 1024 -r 5 -p float

```


Note: for the evaluation of `GEMM`, please use the corresponding program in the `sample/invidual_modules` folder.

