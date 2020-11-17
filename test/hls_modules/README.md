# FBLAS Testing Modules Generator

For testing the modules generator, a set of test routines have been devised.
Each test program is identified with the following name `test_<blas_routine_name>`

The test program for a given routine can be executed by invoking `make test_<blas_routine_name>`
- it will execute the module generator, for creating the module and helper codes
- it will test it

Please note: the majority of the available modules are already tested with the host_api testing unit
