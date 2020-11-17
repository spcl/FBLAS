This folder contains the different program used to derive the CPU performance.

For compiling them:
```
$ make <name_program>_program
```
where `<name_program>` can be: dot, gemv, gemm, axpydot, bicg, gemver, batched_gemm, batched_trsm.


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


Every host program accepts, among the other, the number of runs to execute and produces the averaged execution time,
as well as confidence intervals.


Example, for compiling and evaluate GEMM (squared matrices):

```Bash
# Compile
make gemm_program

# Execute by indicating matrices sizes (n), alpha (a), beta(b) and precision.
bin/gemm_program -n 4096 -a 2 -c 2 -r 10 -p single
```

The program will run for different number of threads (MAX_NUM_THREADS macro defined in the source code) and returns
the lowest execution time and corresponding power consumption.


 