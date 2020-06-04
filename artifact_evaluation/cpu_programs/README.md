This folder contains the different program used to derive the CPU performance.

For compiling them:
```
$ make <name_program>_program
```
wheer `<name_program>` can be: dot, gemv, gemm, axpydot, bicg, gemver, batched_gemm, batched_trsm

They:
 - use Intel MKL Library as BLAS implementation 
 - can work in single/double precision
 - when executed, they run the routine(s) multiple time with different
    number of thread and report the best execution time
 - for power measurement they rely on Mammut (https://github.com/DanieleDeSensi/mammut)
    Power measurement can be disabled by not passing the "POWER_MEASUREMENT"
    macro at the compilation stage
 - the Makefile contains the location of Mammut Library and MKL compilation flag. Please edit
 them according to your setup.   
 