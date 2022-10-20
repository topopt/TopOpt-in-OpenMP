#pragma once

#include "definitions.h"

struct CSRMatrix {
  uint64_t nnz;
  int32_t nrows;

  int *rowOffsets;
  int *colIndex;
  MTYPE *vals;
};

void allocateSubspaceMatrix(const struct gridContext gc, const int l,
                            struct CSRMatrix *M);

void freeSubspaceMatrix(struct CSRMatrix *M);

void assembleSubspaceMatrix(const struct gridContext gc, const int l,
                            const DTYPE *x, struct CSRMatrix M, MTYPE *tmp);

void applyStateOperatorSubspaceMatrix(const struct gridContext gc, const int l,
                                      const struct CSRMatrix M, const CTYPE *in,
                                      CTYPE *out);
