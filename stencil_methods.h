#pragma once

#include "definitions.h"

void applyStateOperator_stencil(const struct gridContext gc, const DTYPE *x,
                                const CTYPE *in, CTYPE *out);

void applyStateOperatorSubspace_halo(const struct gridContext gc, const int l,
                                     const DTYPE *x, CTYPE *in, CTYPE *out);

void getComplianceAndSensetivity_halo(const struct gridContext gc,
                                      const DTYPE *x, STYPE *u, DTYPE *c,
                                      DTYPE *dcdx);

void projectToFinerGrid_halo(const struct gridContext gc,
                             /*in*/ const int l,   /*in*/
                             const CTYPE *ucoarse, /*in*/
                             CTYPE *ufine /*out*/);

void projectToCoarserGrid_halo(const struct gridContext gc,
                               /*in*/ const int l, /*in*/
                               const CTYPE *ufine, /*in*/
                               CTYPE *ucoarse /*out*/);

void assembleInvertedMatrixDiagonalSubspace_halo(const struct gridContext gc,
                                                 const DTYPE *x, const int l,
                                                 MTYPE *diag);
