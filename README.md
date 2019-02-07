<img align="left" width="128" height="128" src="/misc/fblas_logo.png?raw=true">

# FBLAS

**FBLAS** is a porting of the BLAS numerical library ([http://www.netlib.org/blas/](URL)) for Intel FPGA platform. 


## Code

### Requirements

The library relies on:

* Intel FPGA SDK for OpenCL pro, version 18+ ([http://fpgasoftware.intel.com/opencl/](URL))
* GCC (version 5+)
* Rapidjson ([http://rapidjson.org/](URL))
* Google Test (for testing only)

### Installation

After cloning this repository, make sure you clone the [rapidjson](http://rapidjson.org/) submodule dependency, by executing the following command:

```
git submodule update --init
```

After this,the included Makefile can be used to compile code and modules generator.

## The FBLAS library

<img align="right" width="256" height="224" src="/misc/fblas_design.png?raw=true">

FBLAS provides two layers of abtraction: 

* **HLS modules**, which can be integrated into existing hardware designs. They implement BLAS routines (`DOT`, `GEMV`, `GEMM`, etc.). Modules have been designed with compute performance in mind,   exploiting the spatial parallelism and fast on-chip memory on FPGAs and have a streaming in-
terface: data is received and produced using channels;

* a high-level **Host API** conforming to the classical BLAS interface and allows the user to invoke routines directly from an host program. No prior knowledge on FPGA architecture and/or tools is needes. The user writes a standard OpenCL program: she is responsible to transferring data to and from
the device, she can invoke the desired FBLAS routines working on the FPGA memory, and then she copies back the result from the device.

For further information on how to use the library, please refer to: 



## Contact

FBLAS can be used to build numerical applications, and modified to include new features.
Contributions, comments, and issues are welcome!

## License

FBLAS is published under the New BSD license, see [LICENSE](LICENSE).