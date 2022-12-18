#pragma once

#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define __force_inline __attribute__((always_inline))
#define __force_unroll __attribute__((optimize("unroll-loops")))
#define __alignBound 64

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) > (b) ? (b) : (a))

// define stencil sizes at compile time
#define STENCIL_SIZE_X 1 // must be 1
#define STENCIL_SIZE_Y 8 // set to 4 for AVX2, or 8 for AVX512
#define STENCIL_SIZE_Z 1 // must be 1

#define number_of_matrix_free_levels 2

typedef double MTYPE; // matrix type, for stencils, coarse space and such.
typedef float STYPE;  // state type, for displacements forces and such.
typedef float CTYPE;  // cg type, for other arrays in the conjugate gradient
                      // method for the fine level.
typedef float DTYPE;  // design type, for element denseties, gradients and such.

struct FixedDofs {
  uint_fast32_t n;
  uint_fast32_t *idx;
};

struct gridContext {
  double E0;
  double Emin;
  double nu;
  double elementSizeX;
  double elementSizeY;
  double elementSizeZ;
  uint_fast32_t nelx;
  uint_fast32_t nely;
  uint_fast32_t nelz;
  uint_fast32_t wrapx;
  uint_fast32_t wrapy;
  uint_fast32_t wrapz;
  double penal;
  MTYPE **precomputedKE;

  struct FixedDofs *fixedDofs;
};
