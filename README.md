<img align="left" width="128" height="128" src="/misc/fblas_logo.png?raw=true">

# FBLAS

**FBLAS** is a porting of the BLAS numerical library ([http://www.netlib.org/blas/](http://www.netlib.org/blas/)) for Intel FPGA platform. 
For more details, see our [paper](https://arxiv.org/abs/1907.07929).

&nbsp;


## Code

### Requirements

The library depends on:

* Intel FPGA SDK for OpenCL pro, version 18+ ([http://fpgasoftware.intel.com/opencl/](http://fpgasoftware.intel.com/opencl/))
* GCC (version 5+)
* Rapidjson ([http://rapidjson.org/](http://rapidjson.org/))
* Google Test (only for unit tests)

### Installation

After cloning this repository, make sure you clone the [rapidjson](http://rapidjson.org/) submodule dependency, by executing the following command:

```
git submodule update --init
```

After this, the included Makefile can be used to compile code and modules generator:


```
make all
```

## The FBLAS library

<img align="right" width="256" height="220" src="/misc/fblas_design.png?raw=true">

FBLAS provides two layers of abstraction: 

* **HLS modules**, which can be integrated into existing hardware designs. They implement BLAS routines (`DOT`, `GEMV`, `GEMM`, etc.). Modules have been designed with compute performance in mind, exploiting the spatial parallelism and fast on-chip memory on FPGAs and have a streaming interface: data is received and produced using channels. In this way, they can be composed and communicate using on-chip resources rather than off-chip device RAM;

* a high-level **Host API** conforming to the classical BLAS interface that allows the user to invoke routines directly from a host program. No prior knowledge on FPGA architecture and/or tools is needed. The user writes a standard OpenCL program: she is responsible to transferring data to and from
the device, she can invoke the desired FBLAS routines working on the FPGA memory, and then she copies back the result from the device.

For further information on how to use the library, please refer to the [wiki](https://github.com/spcl/FBLAS/wiki).


## Publication
If you use FBLAS, please cite us:
```
@article{
  author={Tiziano De Matteis and Johannes de Fine Licht and Torsten Hoefler},
  title={{FBLAS: Streaming Linear Algebra on FPGA}},
  journal={CoRR},
  year={2019},
  month={Jul.},
  volume={abs/1907.07929},
}
```


## Contact

FBLAS can be used to build numerical applications, and be modified to include new features.
Contributions, comments, and issues are welcome!

## License

FBLAS is published under the New BSD license, see [LICENSE](LICENSE).
