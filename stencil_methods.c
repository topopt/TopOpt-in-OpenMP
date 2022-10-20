#include "stencil_methods.h"

#include "stencil_utility.h"

#include <cblas.h>

void applyStateOperator_stencil(const struct gridContext gc, const DTYPE *x,
                                const CTYPE *in, CTYPE *out) {

  const uint32_t nx = gc.nelx + 1;
  const uint32_t ny = gc.nely + 1;
  const uint32_t nz = gc.nelz + 1;

  // this is necessary for omp to recognize that gc.precomputedKE[0] is already
  // mapped
  const MTYPE *precomputedKE = gc.precomputedKE[0];

  // loop over elements, depends on the which level you are on. For the finest
  // (level 0) nelx*nely*nelz = 100.000 or more, but for every level you go down
  // the number of iterations reduce by a factor of 8. i.e. level 2 will only
  // have ~1000. This specific loop accounts for ~90% runtime
  //#pragma omp teams distribute parallel for collapse(3) schedule(static)

#pragma omp parallel for schedule(static) collapse(3)
  for (int32_t i = 1; i < nx + 1; i += STENCIL_SIZE_X) {
    for (int32_t k = 1; k < nz + 1; k += STENCIL_SIZE_Z) {
      for (int32_t j = 1; j < ny + 1; j += STENCIL_SIZE_Y) {

        alignas(__alignBound) MTYPE out_x[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE out_y[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE out_z[STENCIL_SIZE_Y];

        alignas(__alignBound) MTYPE in_x[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE in_y[STENCIL_SIZE_Y];
        alignas(__alignBound) MTYPE in_z[STENCIL_SIZE_Y];

// zero the values about to be written in this
#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)               \
    aligned(out_x, out_y, out_z                                                \
            : __alignBound)
        for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
          out_x[jj] = 0.0;
          out_y[jj] = 0.0;
          out_z[jj] = 0.0;
        }

        // center line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){0, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){0, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){0, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 0, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, 1, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){0, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){0, -1, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){1, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 0},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 0},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 0},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 0},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // side line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){-1, 0, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, -1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 0},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 0},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, 1, 0}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 0},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 0},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){-1, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, -1},
                                        (const int[]){-1, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, -1},
                                        (const int[]){-1, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){1, 0, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, -1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, -1},
                                        (const int[]){0, -1, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, 1, -1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, -1},
                                        (const int[]){0, 0, -1}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){1, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 0, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, 1, 1},
                                        (const int[]){0, 0, 0}, ny, nz, x, in_x,
                                        in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){1, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){1, -1, 1},
                                        (const int[]){0, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        // edge line, uses uses the same 15 doubles from in
        loadStencilInput(gc, i, j, k, (const int[]){-1, 0, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 0, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, -1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, -1, 1},
                                        (const int[]){-1, -1, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

        loadStencilInput(gc, i, j, k, (const int[]){-1, 1, 1}, in, in_x, in_y,
                         in_z);
        applyStateStencilSpoke_finegrid(gc, precomputedKE, i, j, k,
                                        (const int[]){-1, 1, 1},
                                        (const int[]){-1, 0, 0}, ny, nz, x,
                                        in_x, in_y, in_z, out_x, out_y, out_z);

#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)
        for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {
          const uint_fast32_t offset =
              3 * (i * gc.wrapy * gc.wrapz + k * gc.wrapy + j + jj);
          out[offset + 0] = out_x[jj];
          out[offset + 1] = out_y[jj];
          out[offset + 2] = out_z[jj];
        }
      }
    }
  }

// zero out the extra padded nodes
#pragma omp parallel for collapse(3) schedule(static)
  for (int32_t i = 0; i < gc.wrapx; i++)
    for (int32_t k = 0; k < gc.wrapz; k++)
      for (int32_t j = ny + 1; j < gc.wrapy; j++) {

        const uint_fast32_t offset =
            3 * (i * gc.wrapy * gc.wrapz + k * gc.wrapy + j);

        out[offset + 0] = 0.0;
        out[offset + 1] = 0.0;
        out[offset + 2] = 0.0;
      }

  const uint_fast32_t n = gc.fixedDofs[0].n;
  const uint_fast32_t *fidx = gc.fixedDofs[0].idx;

// apply boundaryConditions
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    out[fidx[i]] = in[fidx[i]];
  }
}

// Apply the global matrix vector product out = K * in
// temperature: very hot, called ~25 x (number of mg levels [1-5]) x
// (number of cg iterations [125-2500]) = [125-12500]  times pr design
// iteration
void applyStateOperatorSubspace_halo(const struct gridContext gc, const int l,
                                     const DTYPE *x, CTYPE *in, CTYPE *out) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int paddingxc =
      (STENCIL_SIZE_X - ((nelxc + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapxc = nelxc + paddingxc + 3;
  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  const uint_fast32_t ndofc = 3 * wrapyc * wrapzc * wrapxc;

#pragma omp parallel for simd schedule(static) safelen(3)
  for (uint32_t i = 0; i < ndofc; i++)
    out[i] = 0.0;

  // loop over elements, depends on the which level you are on. For the finest
  // (level 0) nelx*nely*nelz = 100.000 or more, but for every level you go down
  // the number of iterations reduce by a factor of 8. i.e. level 2 will only
  // have ~1000. This specific loop accounts for ~90% runtime
  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)

#pragma omp parallel for collapse(3) schedule(static)
        for (int32_t i = bx + 1; i < nelxc + 1; i += 2)
          for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
            for (int32_t j = by + 1; j < nelyc + 1; j += 2) {

              alignas(__alignBound) uint_fast32_t edof[24];
              alignas(__alignBound) MTYPE u_local[24];
              alignas(__alignBound) MTYPE out_local[24];

              getEdof_halo(edof, i, j, k, wrapyc, wrapzc);

#pragma omp simd safelen(24) aligned(out_local : __alignBound)
              for (int ii = 0; ii < 24; ii++)
                out_local[ii] = 0.0;

              for (int ii = 0; ii < 24; ii++)
                u_local[ii] = (MTYPE)in[edof[ii]];

              // loop over interior subcells, depends on the level. total
              // iterations = (level+1)^3, i.e. only one iteration for the
              // finest level 0, but inreasing cubicly. Note that the total
              // amount of inner iterations nested by the inner and outer sets
              // of loops is always constant  ( across all levels, that means
              // that as the level number grows, the parallelization available
              // is shifted from the outer loops to the inner loops.
              for (int ii = 0; ii < ncell; ii++)
                for (int kk = 0; kk < ncell; kk++)
                  for (int jj = 0; jj < ncell; jj++) {
                    const int ifine = ((i - 1) * ncell) + ii + 1;
                    const int jfine = ((j - 1) * ncell) + jj + 1;
                    const int kfine = ((k - 1) * ncell) + kk + 1;

                    const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                    const uint_fast32_t elementIndex =
                        ifine * (gc.wrapy - 1) * (gc.wrapz - 1) +
                        kfine * (gc.wrapy - 1) + jfine;
                    const MTYPE elementScale =
                        gc.Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (gc.E0 - gc.Emin);

                    cblas_dgemv(CblasRowMajor, CblasNoTrans, 24, 24,
                                elementScale,
                                gc.precomputedKE[l] + 24 * 24 * cellidx, 24,
                                u_local, 1, 1.0, out_local, 1);
                  }

              for (int iii = 0; iii < 24; iii++)
                out[edof[iii]] += (CTYPE)out_local[iii];
            }

            // apply boundaryConditions
#pragma omp parallel for schedule(static)
  for (int i = 0; i < gc.fixedDofs[l].n; i++) {
    out[gc.fixedDofs[l].idx[i]] = in[gc.fixedDofs[l].idx[i]];
  }
}

// projects a field to a finer grid ucoarse -> ufine
// temperature: medium, called (number of mg levels [1-5]) x (number of cg
// iterations [5-100]) = [5-500]  times pr design iteration
void projectToFinerGrid_halo(const struct gridContext gc,
                             /*in*/ const int l,   /*in*/
                             const CTYPE *ucoarse, /*in*/
                             CTYPE *ufine /*out*/) {

  const int ncellf = pow(2, l);
  const int ncellc = pow(2, l + 1);

  const int nelxf = gc.nelx / ncellf;
  const int nelyf = gc.nely / ncellf;
  const int nelzf = gc.nelz / ncellf;

  const int nelyc = gc.nely / ncellc;
  const int nelzc = gc.nelz / ncellc;

  const int nxf = nelxf + 1;
  const int nyf = nelyf + 1;
  const int nzf = nelzf + 1;

  const int paddingyf =
      (STENCIL_SIZE_Y - ((nelyf + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzf =
      (STENCIL_SIZE_Z - ((nelzf + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyf = nelyf + paddingyf + 3;
  const int wrapzf = nelzf + paddingzf + 3;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more
#pragma omp parallel for collapse(3) schedule(static)
  for (int32_t ifine = 1; ifine < nxf + 1; ifine++)
    for (int32_t kfine = 1; kfine < nzf + 1; kfine++)
      for (int32_t jfine = 1; jfine < nyf + 1; jfine++) {

        const uint32_t fineIndex =
            ifine * wrapyf * wrapzf + kfine * wrapyf + jfine;

        const uint32_t icoarse1 = (ifine - 1) / 2 + 1;
        const uint32_t icoarse2 = (ifine) / 2 + 1;
        const uint32_t jcoarse1 = (jfine - 1) / 2 + 1;
        const uint32_t jcoarse2 = (jfine) / 2 + 1;
        const uint32_t kcoarse1 = (kfine - 1) / 2 + 1;
        const uint32_t kcoarse2 = (kfine) / 2 + 1;

        // Node indices on coarse grid
        const uint_fast32_t coarseIndex1 =
            icoarse1 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex2 =
            icoarse2 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex3 =
            icoarse2 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex4 =
            icoarse1 * wrapyc * wrapzc + kcoarse1 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex5 =
            icoarse1 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex6 =
            icoarse2 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse2;
        const uint_fast32_t coarseIndex7 =
            icoarse2 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse1;
        const uint_fast32_t coarseIndex8 =
            icoarse1 * wrapyc * wrapzc + kcoarse2 * wrapyc + jcoarse1;

        ufine[3 * fineIndex + 0] = 0.125 * ucoarse[3 * coarseIndex1 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex2 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex3 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex4 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex5 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex6 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex7 + 0] +
                                   0.125 * ucoarse[3 * coarseIndex8 + 0];

        ufine[3 * fineIndex + 1] = 0.125 * ucoarse[3 * coarseIndex1 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex2 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex3 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex4 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex5 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex6 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex7 + 1] +
                                   0.125 * ucoarse[3 * coarseIndex8 + 1];

        ufine[3 * fineIndex + 2] = 0.125 * ucoarse[3 * coarseIndex1 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex2 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex3 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex4 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex5 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex6 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex7 + 2] +
                                   0.125 * ucoarse[3 * coarseIndex8 + 2];
      }
}

// projects a field to a coarser grid ufine -> ucoarse
// temperature: medium, called (number of mg levels [1-5]) x (number of cg
// iterations [5-100]) = [5-500]  times pr design iteration
void projectToCoarserGrid_halo(const struct gridContext gc,
                               /*in*/ const int l, /*in*/
                               const CTYPE *ufine, /*in*/
                               CTYPE *ucoarse /*out*/) {

  const int ncellf = pow(2, l);
  const int ncellc = pow(2, l + 1);

  const int nelyf = gc.nely / ncellf;
  const int nelzf = gc.nelz / ncellf;

  const int nelxc = gc.nelx / ncellc;
  const int nelyc = gc.nely / ncellc;
  const int nelzc = gc.nelz / ncellc;

  const int nxc = nelxc + 1;
  const int nyc = nelyc + 1;
  const int nzc = nelzc + 1;

  const int paddingyf =
      (STENCIL_SIZE_Y - ((nelyf + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzf =
      (STENCIL_SIZE_Z - ((nelzf + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyf = nelyf + paddingyf + 3;
  const int wrapzf = nelzf + paddingzf + 3;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  const MTYPE vals[4] = {1.0, 0.5, 0.25, 0.125};

  // loop over nodes, usually very large with nx*ny*nz = 100.000 or more

#pragma omp parallel for collapse(3) schedule(static)
  for (int32_t icoarse = 1; icoarse < nxc + 1; icoarse++)
    for (int32_t kcoarse = 1; kcoarse < nzc + 1; kcoarse++)
      for (int32_t jcoarse = 1; jcoarse < nyc + 1; jcoarse++) {

        const int coarseIndex =
            icoarse * wrapyc * wrapzc + kcoarse * wrapyc + jcoarse;

        // Node indices on fine grid
        const int nx1 = (icoarse - 1) * 2 + 1;
        const int ny1 = (jcoarse - 1) * 2 + 1;
        const int nz1 = (kcoarse - 1) * 2 + 1;

        const int xmin = nx1 - 1;
        const int ymin = ny1 - 1;
        const int zmin = nz1 - 1;

        const int xmax = nx1 + 2;
        const int ymax = ny1 + 2;
        const int zmax = nz1 + 2;

        ucoarse[3 * coarseIndex + 0] = 0.0;
        ucoarse[3 * coarseIndex + 1] = 0.0;
        ucoarse[3 * coarseIndex + 2] = 0.0;

        // this can be done faster by writing out the 27 iterations by hand,
        // do it when necessary.
        for (int32_t ifine = xmin; ifine < xmax; ifine++)
          for (int32_t kfine = zmin; kfine < zmax; kfine++)
            for (int32_t jfine = ymin; jfine < ymax; jfine++) {

              const uint32_t fineIndex =
                  ifine * wrapyf * wrapzf + kfine * wrapyf + jfine;

              const int ind = (nx1 - ifine) * (nx1 - ifine) +
                              (ny1 - jfine) * (ny1 - jfine) +
                              (nz1 - kfine) * (nz1 - kfine);

              ucoarse[3 * coarseIndex + 0] +=
                  vals[ind] * ufine[3 * fineIndex + 0];
              ucoarse[3 * coarseIndex + 1] +=
                  vals[ind] * ufine[3 * fineIndex + 1];
              ucoarse[3 * coarseIndex + 2] +=
                  vals[ind] * ufine[3 * fineIndex + 2];
            }
      }
}

// generate the matrix diagonal for jacobi smoothing.
// temperature: low-medium, called number of levels for every design
// iteration.
void assembleInvertedMatrixDiagonalSubspace_halo(const struct gridContext gc,
                                                 const DTYPE *x, const int l,
                                                 MTYPE *diag) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int paddingxc =
      (STENCIL_SIZE_X - ((nelxc + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapxc = nelxc + paddingxc + 3;
  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  const uint_fast32_t ndofc = 3 * wrapyc * wrapzc * wrapxc;

#pragma omp parallel for simd safelen(3) schedule(static)
  for (unsigned int i = 0; i < ndofc; i++)
    diag[i] = 0.0;

  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)

#pragma omp parallel for collapse(3) schedule(static)
        for (int32_t i = bx + 1; i < nelxc + 1; i += 2)
          for (int32_t k = bz + 1; k < nelzc + 1; k += 2)
            for (int32_t j = by + 1; j < nelyc + 1; j += 2) {

              uint_fast32_t edof[24];
              getEdof_halo(edof, i, j, k, wrapyc, wrapzc);

              for (int ii = 0; ii < ncell; ii++)
                for (int kk = 0; kk < ncell; kk++)
                  for (int jj = 0; jj < ncell; jj++) {
                    const int ifine = ((i - 1) * ncell) + ii + 1;
                    const int jfine = ((j - 1) * ncell) + jj + 1;
                    const int kfine = ((k - 1) * ncell) + kk + 1;

                    const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                    const uint_fast32_t elementIndex =
                        ifine * (gc.wrapy - 1) * (gc.wrapz - 1) +
                        kfine * (gc.wrapy - 1) + jfine;
                    const MTYPE elementScale =
                        gc.Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (gc.E0 - gc.Emin);

                    for (int iii = 0; iii < 24; iii++)
                      diag[edof[iii]] +=
                          elementScale * gc.precomputedKE[l][24 * 24 * cellidx +
                                                             iii * 24 + iii];
                  }
            }

// apply boundaryConditions
#pragma omp parallel for schedule(static)
  for (int i = 0; i < gc.fixedDofs[l].n; i++)
    diag[gc.fixedDofs[l].idx[i]] = 1.0;

#pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i < nelxc + 2; i++)
    for (int k = 1; k < nelzc + 2; k++)
      for (int j = 1; j < nelyc + 2; j++) {
        const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

        diag[3 * nidx + 0] = 1.0 / diag[3 * nidx + 0];
        diag[3 * nidx + 1] = 1.0 / diag[3 * nidx + 1];
        diag[3 * nidx + 2] = 1.0 / diag[3 * nidx + 2];
      }
}

// generate elementwise gradients from displacement.
// temperature: cold, called once for every design iteration.
void getComplianceAndSensetivity_halo(const struct gridContext gc,
                                      const DTYPE *x, STYPE *u, DTYPE *c,
                                      DTYPE *dcdx) {

  c[0] = 0.0;
  DTYPE cc = 0.0;

// loops over all elements, typically 100.000 or more. Note that there are no
// write dependencies, other than the reduction.
#pragma omp parallel for collapse(3) reduction(+ : cc) schedule(static)
  for (int32_t i = 1; i < gc.nelx + 1; i++)
    for (int32_t k = 1; k < gc.nelz + 1; k++)
      for (int32_t j = 1; j < gc.nely + 1; j++) {

        uint_fast32_t edof[24];

        getEdof_halo(edof, i, j, k, gc.wrapy, gc.wrapz);
        const uint_fast32_t elementIndex =
            i * (gc.wrapy - 1) * (gc.wrapz - 1) + k * (gc.wrapy - 1) + j;

        // clocal = ulocal^T * ke * ulocal
        MTYPE clocal = 0.0;
        for (int ii = 0; ii < 24; ii++) {
          for (int jj = 0; jj < 24; jj++) {
            clocal +=
                u[edof[ii]] * gc.precomputedKE[0][24 * ii + jj] * u[edof[jj]];
          }
        }

        // apply contribution to c and dcdx
        cc += clocal * (gc.Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (gc.E0 - gc.Emin));
        dcdx[elementIndex] = clocal * (-3.0 * (gc.E0 - gc.Emin) *
                                       x[elementIndex] * x[elementIndex]);
      }
  c[0] = cc;
}
