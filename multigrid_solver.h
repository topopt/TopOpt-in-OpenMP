#pragma once

#include "definitions.h"

#include "coarse_assembly.h"
#include "coarse_solver.h"

struct SolverData {

  // cg data
  CTYPE *r;
  CTYPE *p;
  CTYPE *q;
  CTYPE *z;

  // jacobi + mg data
  MTYPE **invD;
  CTYPE **dmg;
  CTYPE **rmg;
  CTYPE **zmg;

  // explicitly assembled matrices
  struct CSRMatrix *coarseMatrices;
  struct CoarseSolverData bottomSolver;
};

void solveMultigrid(const struct gridContext gc, DTYPE *x, const int nswp,
                    const int nl, const CTYPE tol, struct SolverData *data,
                    int *finalIter, float *finalRes, CTYPE *b, STYPE *u);

void allocateSolverData(const struct gridContext gc, const int nl,
                        struct SolverData *data);

void freeSolverData(struct SolverData *data, const int nl);
