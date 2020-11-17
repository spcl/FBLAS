This directory contains the programs needed to reproduce the results reported in the "Evaluation" section.

It is organized in the following subfolders:
- `individual_modules`: contains the programs used for evaluation performed in section 6.B;
- `modules_composition`: contains the programs used for evaluation performend in section 6.C and 6.D (axpydot, bicg and gemver examples)
- `cpu_comparison`: contains the programs used for evaluation perfomed in section 6.D (dot, gemv)
- `fully_unrolled`: contains the programs used for evaluation performed in section 6.D (fully unrolled routines)


Each subfolder contains a README with additional and more detailed information, and a Makefile for facilitating compilation.


Requirements (in addition to FBLAS main requirements):
- OpenBLAS (https://www.openblas.net/) for validating the result
- Intel MKL, for measuring the performance of the cpu-only version of the programs (`cpu_comparison`)
- Mammut Library, (https://github.com/DanieleDeSensi/mammut) for measuring power consumption of the cpu-only version of the programs (`cpu_comparison`)



