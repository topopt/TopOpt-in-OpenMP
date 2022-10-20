#pragma once

#include "definitions.h"

void writeDensityVtkFile(const int nelx, const int nely, const int nelz,
                         const DTYPE *densityArray, const char *filename);

void writeDensityVtkFileWithPadding(const int nelx, const int nely,
                                    const int nelz, const DTYPE *densityArray,
                                    const char *filename);
