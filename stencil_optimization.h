#pragma once

#include "definitions.h"

void top3dmgcg(const uint_fast32_t nelx, const uint_fast32_t nely,
               const uint_fast32_t nelz, const DTYPE volfrac, const DTYPE rmin,
               const uint_fast32_t nl, const int design_iters,
               const float cgtol, const uint_fast32_t cgmax);

void applyDensityFilter(const struct gridContext gc, const DTYPE rmin,
                        const DTYPE *rho, DTYPE *out);

void applyDensityFilterGradient(const struct gridContext gc, const DTYPE rmin,
                                DTYPE *v);
