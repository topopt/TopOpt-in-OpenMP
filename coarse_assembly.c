#include "coarse_assembly.h"

#include "stencil_utility.h"

void allocateSubspaceMatrix(const struct gridContext gc, const int l,
                            struct CSRMatrix *M) {

  // compute number of rows
  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;
  const int32_t ndofc = 3 * nxc * nyc * nzc;

  // allocate row offset array
  (*M).nrows = ndofc;
  (*M).rowOffsets = malloc(sizeof(int) * (ndofc + 1));

  // calculate size of rows
  (*M).rowOffsets[0] = 0;

#pragma omp parallel for collapse(3) schedule(static)
  for (int i = 0; i < nxc; i++)
    for (int k = 0; k < nzc; k++)
      for (int j = 0; j < nyc; j++) {
        const int nidx = (i * nyc * nzc + nyc * k + j);

        // add 1 to index to offset the result
        const uint32_t idx1 = 3 * nidx + 0 + 1;
        const uint32_t idx2 = 3 * nidx + 1 + 1;
        const uint32_t idx3 = 3 * nidx + 2 + 1;

        const int32_t xmin = MAX(i - 1, 0);
        const int32_t ymin = MAX(j - 1, 0);
        const int32_t zmin = MAX(k - 1, 0);

        const int32_t xmax = MIN(i + 1, nxc - 1);
        const int32_t ymax = MIN(j + 1, nyc - 1);
        const int32_t zmax = MIN(k + 1, nzc - 1);

        const int32_t localSize =
            3 * (xmax - xmin + 1) * (ymax - ymin + 1) * (zmax - zmin + 1);

        (*M).rowOffsets[idx1] = localSize;
        (*M).rowOffsets[idx2] = localSize;
        (*M).rowOffsets[idx3] = localSize;
      }

  // perform cummulative sum
  for (int i = 1; i < ndofc + 1; i++)
    (*M).rowOffsets[i] += (*M).rowOffsets[i - 1];

  // allocate column and val arrays
  const int nnz = (*M).rowOffsets[ndofc];
  (*M).nnz = nnz;
  (*M).colIndex = malloc(sizeof(int) * nnz);
  (*M).vals = malloc(sizeof(MTYPE) * nnz);

  // populate col array
#pragma omp parallel for collapse(3) schedule(static)
  for (int i = 0; i < nxc; i++)
    for (int k = 0; k < nzc; k++)
      for (int j = 0; j < nyc; j++) {
        const int rowNodeIndex = i * nyc * nzc + nyc * k + j;

        // add 1 to index to offset the result
        const uint32_t row1 = 3 * rowNodeIndex + 0;
        const uint32_t row2 = 3 * rowNodeIndex + 1;
        const uint32_t row3 = 3 * rowNodeIndex + 2;

        const int32_t xmin = MAX(i - 1, 0);
        const int32_t ymin = MAX(j - 1, 0);
        const int32_t zmin = MAX(k - 1, 0);

        const int32_t xmax = MIN(i + 2, nxc);
        const int32_t ymax = MIN(j + 2, nyc);
        const int32_t zmax = MIN(k + 2, nzc);

        int rowOffset1 = (*M).rowOffsets[row1];
        int rowOffset2 = (*M).rowOffsets[row2];
        int rowOffset3 = (*M).rowOffsets[row3];

        for (int ii = xmin; ii < xmax; ii++)
          for (int kk = zmin; kk < zmax; kk++)
            for (int jj = ymin; jj < ymax; jj++) {

              const int colNodeIndex = ii * nyc * nzc + nyc * kk + jj;

              const uint32_t col1 = 3 * colNodeIndex + 0;
              const uint32_t col2 = 3 * colNodeIndex + 1;
              const uint32_t col3 = 3 * colNodeIndex + 2;

              (*M).colIndex[rowOffset1 + 0] = col1;
              (*M).colIndex[rowOffset2 + 0] = col1;
              (*M).colIndex[rowOffset3 + 0] = col1;

              (*M).colIndex[rowOffset1 + 1] = col2;
              (*M).colIndex[rowOffset2 + 1] = col2;
              (*M).colIndex[rowOffset3 + 1] = col2;

              (*M).colIndex[rowOffset1 + 2] = col3;
              (*M).colIndex[rowOffset2 + 2] = col3;
              (*M).colIndex[rowOffset3 + 2] = col3;

              rowOffset1 += 3;
              rowOffset2 += 3;
              rowOffset3 += 3;
            }
      }
}

void freeSubspaceMatrix(struct CSRMatrix *M) {
  free((*M).rowOffsets);
  free((*M).vals);
  free((*M).colIndex);
}

void assembleSubspaceMatrix(const struct gridContext gc, const int l,
                            const DTYPE *x, struct CSRMatrix M, MTYPE *tmp) {

// zero the val array
#pragma omp parallel for simd schedule(static) safelen(3)
  for (int i = 0; i < M.nnz; i++)
    M.vals[i] = 0.0;

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  // loop over active elements, in 8 colored fashion to avoid data-races
  for (int32_t bx = 0; bx < 2; bx++)
    for (int32_t bz = 0; bz < 2; bz++)
      for (int32_t by = 0; by < 2; by++)

#pragma omp parallel for collapse(3) schedule(static)
        for (int32_t i = bx; i < nelxc; i += 2)
          for (int32_t k = bz; k < nelzc; k += 2)
            for (int32_t j = by; j < nelyc; j += 2) {

              //// get 'true' edof
              uint_fast32_t edof[24];
              getEdof_halo(edof, i, j, k, nyc, nzc);

              const int32_t i_halo = i + 1;
              const int32_t j_halo = j + 1;
              const int32_t k_halo = k + 1;

              // make local zero size buffer
              alignas(__alignBound) MTYPE ke[24][24];
              for (int iii = 0; iii < 24; iii++)
                for (int jjj = 0; jjj < 24; jjj++)
                  ke[iii][jjj] = 0.0;

              // assemble local matrix
              for (int ii = 0; ii < ncell; ii++)
                for (int kk = 0; kk < ncell; kk++)
                  for (int jj = 0; jj < ncell; jj++) {
                    const int ifine = ((i_halo - 1) * ncell) + ii + 1;
                    const int jfine = ((j_halo - 1) * ncell) + jj + 1;
                    const int kfine = ((k_halo - 1) * ncell) + kk + 1;

                    const int cellidx = ncell * ncell * ii + ncell * kk + jj;

                    const uint_fast32_t elementIndex =
                        ifine * (gc.wrapy - 1) * (gc.wrapz - 1) +
                        kfine * (gc.wrapy - 1) + jfine;
                    const MTYPE elementScale =
                        gc.Emin + x[elementIndex] * x[elementIndex] *
                                      x[elementIndex] * (gc.E0 - gc.Emin);

                    for (int iii = 0; iii < 24; iii++)
                      for (int jjj = 0; jjj < 24; jjj++)
                        ke[iii][jjj] += elementScale *
                                        gc.precomputedKE[l][24 * 24 * cellidx +
                                                            24 * iii + jjj];
                  }

              // add matrix contribution to val slices
              // initial naive implementation w. linear search and row-by-row
              // update
              // potential optimizations: bisection search, node-by-node
              // update
              for (int iii = 0; iii < 24; iii++) {
                const int32_t rowStart = M.rowOffsets[edof[iii]];
                const int32_t rowEnd = M.rowOffsets[edof[iii] + 1];

                for (int jjj = 0; jjj < 24; jjj++) {
                  // printf("Looking for %i in %i, [%i %i]: ", edof[jjj],
                  //        edof[iii], rowStart, rowEnd);

                  for (int idx = rowStart; idx < rowEnd; idx++) {
                    if (M.colIndex[idx] == edof[jjj]) {
                      M.vals[idx] += ke[iii][jjj];
                      // printf("%i", idx);
                      break;
                    }
                  }

                  // printf("\n");
                }
              }
            }

// apply boundaryConditions to the matrix
#pragma omp parallel for schedule(static)
  for (int row = 0; row < M.nrows; row++) {
    tmp[row] = 1.0;
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < gc.fixedDofs[l].n; i++) {
    const int row =
        haloToTrue(gc.fixedDofs[l].idx[i], wrapyc, wrapzc, nyc, nzc);
    tmp[row] = 0.0;
  }

  // if either row or col is marked as fixed dof, zero the term
#pragma omp parallel for schedule(static)
  for (int row = 0; row < M.nrows; row++) {
    const int32_t rowStart = M.rowOffsets[row];
    const int32_t rowEnd = M.rowOffsets[row + 1];

    for (int col = rowStart; col < rowEnd; col++)
      M.vals[col] *= tmp[row] * tmp[M.colIndex[col]];
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < gc.fixedDofs[l].n; i++) {
    const int row =
        haloToTrue(gc.fixedDofs[l].idx[i], wrapyc, wrapzc, nyc, nzc);

    // printf("%i -> %i (%i)\n", gc.fixedDofs[l].idx[i], row, fdtmp.idx[i]);
    const int32_t rowStart = M.rowOffsets[row];
    const int32_t rowEnd = M.rowOffsets[row + 1];
    for (int col = rowStart; col < rowEnd; col++)
      if (row == M.colIndex[col])
        M.vals[col] = 1.0;
  }
}

void applyStateOperatorSubspaceMatrix(const struct gridContext gc, const int l,
                                      const struct CSRMatrix M, const CTYPE *in,
                                      CTYPE *out) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int paddingxc =
      (STENCIL_SIZE_X - ((nelxc + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapxc = nelxc + paddingxc + 3;
  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  const int ndofc = 3 * wrapxc * wrapyc * wrapzc;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndofc; i++)
    out[i] = 0.0;

#pragma omp parallel for collapse(3) schedule(static)
  for (int32_t i = 1; i < nxc + 1; i++)
    for (int32_t k = 1; k < nzc + 1; k++)
      for (int32_t j = 1; j < nyc + 1; j++) {

        const int haloNodeIndex = i * wrapyc * wrapzc + wrapyc * k + j;
        const uint32_t rowHaloIndex1 = 3 * haloNodeIndex + 0;
        const uint32_t rowHaloIndex2 = 3 * haloNodeIndex + 1;
        const uint32_t rowHaloIndex3 = 3 * haloNodeIndex + 2;

        const uint32_t i_no_halo = i - 1;
        const uint32_t j_no_halo = j - 1;
        const uint32_t k_no_halo = k - 1;

        const int rowNodeIndex =
            i_no_halo * nyc * nzc + nyc * k_no_halo + j_no_halo;
        const uint32_t rowIndex1 = 3 * rowNodeIndex + 0;
        const uint32_t rowIndex2 = 3 * rowNodeIndex + 1;
        const uint32_t rowIndex3 = 3 * rowNodeIndex + 2;

        double outBufferRow1 = 0.0;
        double outBufferRow2 = 0.0;
        double outBufferRow3 = 0.0;

        const int32_t xmin = MAX(i - 1, 1);
        const int32_t ymin = MAX(j - 1, 1);
        const int32_t zmin = MAX(k - 1, 1);

        const int32_t xmax = MIN(i + 2, nxc + 1);
        const int32_t ymax = MIN(j + 2, nyc + 1);
        const int32_t zmax = MIN(k + 2, nzc + 1);

        // recalculate indices instead of recreating i,j,k without halo
        int offsetRow1 = M.rowOffsets[rowIndex1];
        int offsetRow2 = M.rowOffsets[rowIndex2];
        int offsetRow3 = M.rowOffsets[rowIndex3];

        for (int ii = xmin; ii < xmax; ii++)
          for (int kk = zmin; kk < zmax; kk++)
            for (int jj = ymin; jj < ymax; jj++) {

              const int haloColNodeIndex =
                  ii * wrapyc * wrapzc + wrapyc * kk + jj;

              const uint32_t colHaloIndex1 = 3 * haloColNodeIndex + 0;
              const uint32_t colHaloIndex2 = 3 * haloColNodeIndex + 1;
              const uint32_t colHaloIndex3 = 3 * haloColNodeIndex + 2;

              outBufferRow1 +=
                  (double)in[colHaloIndex1] * M.vals[offsetRow1 + 0] +
                  (double)in[colHaloIndex2] * M.vals[offsetRow1 + 1] +
                  (double)in[colHaloIndex3] * M.vals[offsetRow1 + 2];

              outBufferRow2 +=
                  (double)in[colHaloIndex1] * M.vals[offsetRow2 + 0] +
                  (double)in[colHaloIndex2] * M.vals[offsetRow2 + 1] +
                  (double)in[colHaloIndex3] * M.vals[offsetRow2 + 2];

              outBufferRow3 +=
                  (double)in[colHaloIndex1] * M.vals[offsetRow3 + 0] +
                  (double)in[colHaloIndex2] * M.vals[offsetRow3 + 1] +
                  (double)in[colHaloIndex3] * M.vals[offsetRow3 + 2];

              offsetRow1 += 3;
              offsetRow2 += 3;
              offsetRow3 += 3;
            }

        out[rowHaloIndex1] = outBufferRow1;
        out[rowHaloIndex2] = outBufferRow2;
        out[rowHaloIndex3] = outBufferRow3;
      }
}
