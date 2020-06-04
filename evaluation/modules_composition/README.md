This directory contains the programs used to evaluate streaming composition of 
different modules.

It is organized in two folders:
- `non_streaming`, contains the version of the programs realized with standard Host API.
    In this case, routines do not stream data each other and resort to DRAM to save data.
- `streaming`, contains the version of the programs realized with streaming composition.
    
For both cases, in the folders there are present Makefile for generating emulation bitstream,
hardware and host programs.

Being `<test_name>` in `{saxpydot, daxpydot, sbicg, dbicg, sgemver, dver}`, the following 
commands can be used:
 - compile emulation `make <test_name>_emulator`
 - compile host program `make axpydot/bicg/gemver_host`
 - synthesize hardware `make <test_name>_hardware`
 
To execute a program:
- if in emulation, set the environment variable ` CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1`
- execute the host program locate under `bin/` by passing the suggested parameters. Note: for non-streamed
    version it will require the json file automatically generated and located under `<test_name>codegen_files/generated_routines.json`
 

Every host program accepts, among the other, the number of runs to execute and produces the averaged execution time,
as well as confidence intervals.
The result is validated against openblas (CPU) version of the same computation.