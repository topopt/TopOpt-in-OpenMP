#pragma once

#include "definitions.h"

__force_inline int inline getLocalNodeIndex(
    const int nodeOffsetFromElement[3]) {
  int output = -1;

  const int ox = nodeOffsetFromElement[0];
  const int oy = nodeOffsetFromElement[1];
  const int oz = nodeOffsetFromElement[2];

  // hack by jumptable, inputs are compile-time constant, so it should be fine.
  // I do miss templates for this stuff though..

  if (ox == 1 && oy == 1 && oz == 1)
    output = 5;
  else if (ox == 1 && oy == 0 && oz == 1)
    output = 6;
  else if (ox == 0 && oy == 1 && oz == 1)
    output = 4;
  else if (ox == 0 && oy == 0 && oz == 1)
    output = 7;
  else if (ox == 1 && oy == 1 && oz == 0)
    output = 1;
  else if (ox == 1 && oy == 0 && oz == 0)
    output = 2;
  else if (ox == 0 && oy == 1 && oz == 0)
    output = 0;
  else if (ox == 0 && oy == 0 && oz == 0)
    output = 3;

  return output;
}

__force_inline inline void applyStateStencilSpoke_finegrid(
    const struct gridContext gc, const MTYPE *precomputedKE, const int i_center,
    const int j_center, const int k_center, const int nodeOffset[3],
    const int elementOffset[3], const uint32_t ny, const uint32_t nz,
    const DTYPE *x, MTYPE inBuffer_x[STENCIL_SIZE_Y],
    MTYPE inBuffer_y[STENCIL_SIZE_Y], MTYPE inBuffer_z[STENCIL_SIZE_Y],
    MTYPE outBuffer_x[STENCIL_SIZE_Y], MTYPE outBuffer_y[STENCIL_SIZE_Y],
    MTYPE outBuffer_z[STENCIL_SIZE_Y]) {

  // compute sending and recieving local node number, hopefully be evaluated at
  // compile-time

  const int recievingNodeOffset[3] = {-elementOffset[0], -elementOffset[1],
                                      -elementOffset[2]};
  const int nodeOffsetFromElement[3] = {nodeOffset[0] - elementOffset[0],
                                        nodeOffset[1] - elementOffset[1],
                                        nodeOffset[2] - elementOffset[2]};

  const int localRecievingNodeIdx = getLocalNodeIndex(recievingNodeOffset);
  const int localSendingNodeIdx = getLocalNodeIndex(nodeOffsetFromElement);

  // compute index for element
  const uint_fast32_t i_element = i_center + elementOffset[0];
  const uint_fast32_t j_element = j_center + elementOffset[1];
  const uint_fast32_t k_element = k_center + elementOffset[2];

  MTYPE localBuf_x[STENCIL_SIZE_Y];
  MTYPE localBuf_y[STENCIL_SIZE_Y];
  MTYPE localBuf_z[STENCIL_SIZE_Y];

  const int startRecieve_local = 3 * localRecievingNodeIdx;
  const int startSend_local = 3 * localSendingNodeIdx;

  // loop over simd stencil size
// currently does not compile to simd instructions..
#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y) aligned(      \
    inBuffer_x, inBuffer_y, inBuffer_z, outBuffer_x, outBuffer_y, outBuffer_z  \
    : __alignBound)
  for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {

    // local coordinates
    const uint_fast32_t elementIndex =
        (i_element) * (gc.wrapy - 1) * (gc.wrapz - 1) +
        (k_element) * (gc.wrapy - 1) + (j_element + jj);
    MTYPE elementScale = gc.Emin + x[elementIndex] * x[elementIndex] *
                                       x[elementIndex] * (gc.E0 - gc.Emin);

    // important, sets true zero to halo values. This is necessary for
    // correctness. Performance can be gained by removing the constant Emin, and
    // setting the minimum allowed density to a corresponding non-zero value.
    // But this is left for the future at the moment.
    if (i_element == 0 || i_element > gc.nelx || j_element + jj == 0 ||
        j_element + jj > gc.nely || k_element == 0 || k_element > gc.nelz)
      elementScale = 0.0;

    localBuf_x[jj] = 0.0;
    localBuf_y[jj] = 0.0;
    localBuf_z[jj] = 0.0;

    // add the spoke contribution
    localBuf_x[jj] +=
        precomputedKE[24 * (startRecieve_local + 0) + (startSend_local + 0)] *
        inBuffer_x[jj];
    localBuf_x[jj] +=
        precomputedKE[24 * (startRecieve_local + 0) + (startSend_local + 1)] *
        inBuffer_y[jj];
    localBuf_x[jj] +=
        precomputedKE[24 * (startRecieve_local + 0) + (startSend_local + 2)] *
        inBuffer_z[jj];

    localBuf_y[jj] +=
        precomputedKE[24 * (startRecieve_local + 1) + (startSend_local + 0)] *
        inBuffer_x[jj];
    localBuf_y[jj] +=
        precomputedKE[24 * (startRecieve_local + 1) + (startSend_local + 1)] *
        inBuffer_y[jj];
    localBuf_y[jj] +=
        precomputedKE[24 * (startRecieve_local + 1) + (startSend_local + 2)] *
        inBuffer_z[jj];

    localBuf_z[jj] +=
        precomputedKE[24 * (startRecieve_local + 2) + (startSend_local + 0)] *
        inBuffer_x[jj];
    localBuf_z[jj] +=
        precomputedKE[24 * (startRecieve_local + 2) + (startSend_local + 1)] *
        inBuffer_y[jj];
    localBuf_z[jj] +=
        precomputedKE[24 * (startRecieve_local + 2) + (startSend_local + 2)] *
        inBuffer_z[jj];

    outBuffer_x[jj] += elementScale * localBuf_x[jj];
    outBuffer_y[jj] += elementScale * localBuf_y[jj];
    outBuffer_z[jj] += elementScale * localBuf_z[jj];
  }
}

__force_inline inline void
loadStencilInput(const struct gridContext gc, const int i_center,
                 const int j_center, const int k_center,
                 const int nodeOffset[3], const CTYPE *in,
                 MTYPE buffer_x[STENCIL_SIZE_Y], MTYPE buffer_y[STENCIL_SIZE_Y],
                 MTYPE buffer_z[STENCIL_SIZE_Y]) {

  const int i_sender = i_center + nodeOffset[0];
  const int j_sender = j_center + nodeOffset[1];
  const int k_sender = k_center + nodeOffset[2];

#pragma omp simd safelen(STENCIL_SIZE_Y) simdlen(STENCIL_SIZE_Y)               \
    aligned(buffer_x, buffer_y, buffer_z                                       \
            : __alignBound)
  for (int jj = 0; jj < STENCIL_SIZE_Y; jj++) {

    const uint_fast32_t sendingNodeIndex =
        (i_sender)*gc.wrapy * gc.wrapz + (k_sender)*gc.wrapy + (j_sender + jj);

    const int startSend = 3 * sendingNodeIndex;

    buffer_x[jj] = in[startSend + 0];
    buffer_y[jj] = in[startSend + 1];
    buffer_z[jj] = in[startSend + 2];
  }
}

// compute indices of displacement for a given element number
// temperature: very very hot, called as part of the hot kernels in the
// program, should be inlined always.
__force_inline inline void getEdof_halo(uint_fast32_t edof[24], const int i,
                                        const int j, const int k,
                                        const int wrapy, const int wrapz) {

  const int nx_1 = i;
  const int nx_2 = i + 1;
  const int nz_1 = k;
  const int nz_2 = k + 1;
  const int ny_1 = j;
  const int ny_2 = j + 1;

  const uint_fast32_t nIndex1 = nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_2;
  const uint_fast32_t nIndex2 = nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_2;
  const uint_fast32_t nIndex3 = nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_1;
  const uint_fast32_t nIndex4 = nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_1;
  const uint_fast32_t nIndex5 = nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_2;
  const uint_fast32_t nIndex6 = nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_2;
  const uint_fast32_t nIndex7 = nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_1;
  const uint_fast32_t nIndex8 = nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_1;

  edof[0] = 3 * nIndex1 + 0;
  edof[1] = 3 * nIndex1 + 1;
  edof[2] = 3 * nIndex1 + 2;
  edof[3] = 3 * nIndex2 + 0;
  edof[4] = 3 * nIndex2 + 1;
  edof[5] = 3 * nIndex2 + 2;
  edof[6] = 3 * nIndex3 + 0;
  edof[7] = 3 * nIndex3 + 1;
  edof[8] = 3 * nIndex3 + 2;
  edof[9] = 3 * nIndex4 + 0;
  edof[10] = 3 * nIndex4 + 1;
  edof[11] = 3 * nIndex4 + 2;

  edof[12] = 3 * nIndex5 + 0;
  edof[13] = 3 * nIndex5 + 1;
  edof[14] = 3 * nIndex5 + 2;
  edof[15] = 3 * nIndex6 + 0;
  edof[16] = 3 * nIndex6 + 1;
  edof[17] = 3 * nIndex6 + 2;
  edof[18] = 3 * nIndex7 + 0;
  edof[19] = 3 * nIndex7 + 1;
  edof[20] = 3 * nIndex7 + 2;
  edof[21] = 3 * nIndex8 + 0;
  edof[22] = 3 * nIndex8 + 1;
  edof[23] = 3 * nIndex8 + 2;
}

// convert the node index from coordinates with halo padding to a grid without.
// requires the wrapping parameters from the grid with halo and size of the grid
// without
__force_inline inline int haloToTrue(const int index, const int wrapy,
                                     const int wrapz, const int ny,
                                     const int nz) {

  const int nodeNumber = index / 3;
  const int dofOffset = index % 3;

  const int i_halo = nodeNumber / (wrapy * wrapz);
  const int j_halo = nodeNumber % wrapy;
  const int k_halo = (nodeNumber % (wrapy * wrapz)) / wrapy;

  const int i = i_halo - 1;
  const int j = j_halo - 1;
  const int k = k_halo - 1;

  const int newNodeNumber = i * ny * nz + k * ny + j;

  return 3 * newNodeNumber + dofOffset;
}
