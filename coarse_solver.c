#include "coarse_solver.h"

void initializeCoarseSolver(const struct gridContext gc, const int l,
                            struct CoarseSolverData *solverData,
                            const struct CSRMatrix M) {

  solverData->cholmodCommon = malloc(sizeof(cholmod_common));

  cholmod_start(solverData->cholmodCommon);

  solverData->cholmodCommon->nmethods = 9;

  solverData->sparseMatrix = cholmod_allocate_sparse(
      M.nrows, /* # of rows of A */
      M.nrows, /* # of columns of A */
      M.nnz,   /* max # of nonzeros of A */
      1,       /* TRUE if columns of A sorted, FALSE otherwise */
      1,       /* TRUE if A will be packed, FALSE otherwise */
      1, /* stype of A 0=use both upper and lower, 1=use upper, -1 use lower */
      CHOLMOD_REAL, /* CHOLMOD_PATTERN, _REAL, _COMPLEX, or _ZOMPLEX */
      solverData->cholmodCommon);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < M.nrows + 1; i++)
    ((int *)solverData->sparseMatrix->p)[i] = M.rowOffsets[i];

#pragma omp parallel for schedule(static)
  for (int i = 0; i < M.nnz; i++)
    ((int *)solverData->sparseMatrix->i)[i] = M.colIndex[i];

  solverData->factoredMatrix = cholmod_analyze(
      solverData->sparseMatrix, /* matrix to order and analyze */
      solverData->cholmodCommon);

  solverData->rhs = cholmod_allocate_dense(
      M.nrows,      /* # of rows of matrix */
      1,            /* # of cols of matrix */
      M.nrows,      /* leading dimension */
      CHOLMOD_REAL, /* CHOLMOD_REAL, _COMPLEX, or _ZOMPLEX */
      solverData->cholmodCommon);

  solverData->solution = NULL;
  solverData->Y_workspace = NULL;
  solverData->E_workspace = NULL;
}

void freeCoarseSolver(const struct gridContext gc, const int l,
                      struct CoarseSolverData *solverData,
                      const struct CSRMatrix M) {

  // cholmod_print_common("common", solverData->cholmodCommon);

  cholmod_free_dense(&solverData->rhs, solverData->cholmodCommon);
  cholmod_free_dense(&solverData->solution, solverData->cholmodCommon);
  cholmod_free_dense(&solverData->Y_workspace, solverData->cholmodCommon);
  cholmod_free_dense(&solverData->E_workspace, solverData->cholmodCommon);

  cholmod_free_sparse(&solverData->sparseMatrix, solverData->cholmodCommon);
  cholmod_free_factor(&solverData->factoredMatrix, solverData->cholmodCommon);

  cholmod_finish(solverData->cholmodCommon);
  free(solverData->cholmodCommon);
}

void factorizeSubspaceMatrix(const struct gridContext gc, const int l,
                             struct CoarseSolverData solverData,
                             const struct CSRMatrix M) {

#pragma omp parallel for
  for (int i = 0; i < M.nnz; i++)
    ((double *)solverData.sparseMatrix->x)[i] = M.vals[i];

  cholmod_factorize(solverData.sparseMatrix, solverData.factoredMatrix,
                    solverData.cholmodCommon);
}

void solveSubspaceMatrix(const struct gridContext gc, const int l,
                         struct CoarseSolverData solverData, const CTYPE *in,
                         CTYPE *out) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int32_t nxc = nelxc + 1;
  const int32_t nyc = nelyc + 1;
  const int32_t nzc = nelzc + 1;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

// copy grid data to vector
#pragma omp for collapse(3)
  for (int i = 1; i < nxc + 1; i++)
    for (int k = 1; k < nzc + 1; k++)
      for (int j = 1; j < nyc + 1; j++) {
        const int nidx = ((i - 1) * nyc * nzc + (k - 1) * nyc + (j - 1));
        const int nidx_s = (i * wrapyc * wrapzc + wrapyc * k + j);

        ((double *)solverData.rhs->x)[3 * nidx + 0] = in[3 * nidx_s + 0];
        ((double *)solverData.rhs->x)[3 * nidx + 1] = in[3 * nidx_s + 1];
        ((double *)solverData.rhs->x)[3 * nidx + 2] = in[3 * nidx_s + 2];
      }

  cholmod_solve2(CHOLMOD_A,                 /* system to solve */
                 solverData.factoredMatrix, /* factorization to use */
                 solverData.rhs,            /* right-hand-side */
                 NULL,                      /* handle */
                 &solverData.solution,      /* solution, allocated if need be */
                 NULL,                      /* handle*/
                 &solverData.Y_workspace,   /* workspace, or NULL */
                 &solverData.E_workspace,   /* workspace, or NULL */
                 solverData.cholmodCommon);

// copy data back to grid format
#pragma omp for collapse(3)
  for (int i = 1; i < nxc + 1; i++)
    for (int k = 1; k < nzc + 1; k++)
      for (int j = 1; j < nyc + 1; j++) {
        const int nidx = ((i - 1) * nyc * nzc + (k - 1) * nyc + (j - 1));
        const int nidx_s = (i * wrapyc * wrapzc + wrapyc * k + j);

        out[3 * nidx_s + 0] = ((double *)solverData.solution->x)[3 * nidx + 0];
        out[3 * nidx_s + 1] = ((double *)solverData.solution->x)[3 * nidx + 1];
        out[3 * nidx_s + 2] = ((double *)solverData.solution->x)[3 * nidx + 2];
      }
}
