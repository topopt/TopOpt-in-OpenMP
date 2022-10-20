#pragma once

#include "definitions.h"

#include "coarse_assembly.h"

#include "cholmod.h"

struct CoarseSolverData {
  cholmod_common *cholmodCommon;
  cholmod_sparse *sparseMatrix;
  cholmod_factor *factoredMatrix;

  cholmod_dense *rhs;
  cholmod_dense *solution;

  cholmod_dense *Y_workspace;
  cholmod_dense *E_workspace;
};

void initializeCoarseSolver(const struct gridContext gc, const int l,
                            struct CoarseSolverData *ssolverData,
                            const struct CSRMatrix M);

void freeCoarseSolver(const struct gridContext gc, const int l,
                      struct CoarseSolverData *solverData,
                      const struct CSRMatrix M);

void factorizeSubspaceMatrix(const struct gridContext gc, const int l,
                             struct CoarseSolverData solverData,
                             const struct CSRMatrix M);

void solveSubspaceMatrix(const struct gridContext gc, const int l,
                         struct CoarseSolverData solverData, const CTYPE *in,
                         CTYPE *out);