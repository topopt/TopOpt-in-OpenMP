[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7791822.svg)](https://doi.org/10.5281/zenodo.7791822)
# Topology Optimisation Based on OpenMP
This repository contains an implementation of a topology optimisation solver for linear elastic compliance minimisation in three dimensions. The implementation is matrix-free and based on OpenMP for multithreading and SIMD support.

## Compilation
The code is tested with GCC 11.2 and the built-in compiler support for OpenMP. 

The following dependencies are also used
| **Pachage**           | **Version** | **Installation** |
| :---                  | :---        | :---           |
| `SuiteSparse/CHOLMOD` | 5.10.1       | [See Github release notes](https://github.com/DrTimothyAldenDavis/SuiteSparse/) |

## Set appropriate SIMD sizes
To improve the performance of the code, the proper stencil width should be set for the current architecture in the definitions header. 

This is done by setting
```c
#define STENCIL_SIZE_Y 4 // set to 4 for AVX2, or 8 for AVX512
```
to either 4 or 8, depending on whether AVX2 or AVX512 is available on the target platform. If in doubt, the default setting should work sufficiently well everywhere.

## Running the Code
The default design problem is a 2x1x1 cantilever problem. To run 20 iterations of the code on a grid of 128 times 64 times 64 voxels using 12 cores the following commands are used:
```bash
$ export OMP_NUM_THREADS=12
$ export GOMP_CPU_AFFINITY=0-11
$ export CHOLMOD_OMP_NUM_THREADS=12
$ ./top3d -x 16 -y 8 -z 8 -l 4 -i 20
```
In the above example, we specify that we wish to use four levels in the multi-grid preconditioner with `-l 4`. Therefore we must divide the domain size by 2^l-1 to find the size of the coarsest grid which the application takes as input. 

A list of available options is printed on start of the program.

## Authorship
This code has been developed by Erik Albert Träff under the supervision of Niels Aage and Ole Signmund.
