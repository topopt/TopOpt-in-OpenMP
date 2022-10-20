#include <stdbool.h>

#include "stencil_optimization.h"

#include "coarse_solver.h"
#include "grid_utilities.h"
#include "multigrid_solver.h"
#include "stencil_methods.h"
#include "write_vtk.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// main function
void top3dmgcg(const uint_fast32_t nelx, const uint_fast32_t nely,
               const uint_fast32_t nelz, const DTYPE volfrac, const DTYPE rmin,
               const uint_fast32_t nl, const int design_iters,
               const float cgtol, const uint_fast32_t cgmax) {

  struct gridContext gridContext;
  gridContext.E0 = 1;
  gridContext.Emin = 1e-6;
  gridContext.nu = 0.3;
  gridContext.nelx = nelx;
  gridContext.nely = nely;
  gridContext.nelz = nelz;
  gridContext.penal = 3; // dummy variable, does nothing

  gridContext.elementSizeX = 0.5;
  gridContext.elementSizeY = 0.5;
  gridContext.elementSizeZ = 0.5;

  initializeGridContext(&gridContext, nl);

  const uint_fast64_t nelem = (gridContext.wrapx - 1) *
                              (gridContext.wrapy - 1) * (gridContext.wrapz - 1);

  CTYPE *F;
  STYPE *U;
  allocateStateField(gridContext, 0, &F);
  allocateStateField_STYPE(gridContext, 0, &U);

  // for (int i = 1; i < gridContext.nelx + 2; i++)
  // for (int k = 1; k < gridContext.nelz + 2; k++)
  double forceMagnitude = -1;

  { // setup cantilever load
    const int ny = nely + 1;

    const int k = 0;

    const double radius = ((double)ny) / 5.0; // snap
    const double radius2 = radius * radius;
    const double center_x = (double)nelx;
    const double center_y = ((double)nely - 1.0) / 2.0;

    int num_elements = 0;
    for (int i = 0; i < nelx; i++) {
      for (int j = 0; j < nely; j++) {
        const double dx = (double)i - center_x;
        const double dy = (double)j - center_y;
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < radius2) {
          num_elements++;
        }
      }
    }

    double nodalForce = forceMagnitude / (4.0 * (double)num_elements);
    for (int i = 0; i < nelx; i++) {
      for (int j = 0; j < nely; j++) {

        const int ii = i + 1;
        const int jj = j + 1;
        const int kk = k + 1;

        const double dx = (double)i - center_x;
        const double dy = (double)j - center_y;
        const double dist2 = dx * dx + dy * dy;

        if (dist2 < radius2) {
          const uint_fast32_t nidx1 =
              (ii + 1) * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + (jj + 1);
          const uint_fast32_t nidx2 =
              (ii + 1) * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + jj;
          const uint_fast32_t nidx3 =
              ii * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + (jj + 1);
          const uint_fast32_t nidx4 =
              ii * gridContext.wrapy * gridContext.wrapz +
              gridContext.wrapy * kk + jj;
          F[3 * nidx1 + 2] += nodalForce;
          F[3 * nidx2 + 2] += nodalForce;
          F[3 * nidx3 + 2] += nodalForce;
          F[3 * nidx4 + 2] += nodalForce;
        }
      }
    }
  }

  // #pragma omp parallel for
  //   for (int j = 1; j < gridContext.nely + 2; j++) {
  //     const int i = gridContext.nelx;
  //     const int k = 1;

  //     const uint_fast32_t nidx =
  //         i * gridContext.wrapy * gridContext.wrapz + gridContext.wrapy * k +
  //         j;
  //     F[3 * nidx + 2] = -1.0;
  //   }

  //   F[3 * (gridContext.nelx * gridContext.wrapy * gridContext.wrapz +
  //          gridContext.wrapy + 1) +
  //     2] = -0.5;
  //   F[3 * (gridContext.nelx * gridContext.wrapy * gridContext.wrapz +
  //          gridContext.wrapy + (gridContext.nely + 1)) +
  //     2] = -0.5;

  DTYPE *dc = malloc(sizeof(DTYPE) * nelem);
  DTYPE *dv = malloc(sizeof(DTYPE) * nelem);
  DTYPE *x = malloc(sizeof(DTYPE) * nelem);
  DTYPE *xPhys = malloc(sizeof(DTYPE) * nelem);
  DTYPE *xnew = malloc(sizeof(DTYPE) * nelem);
  DTYPE c = 0.0;

#pragma omp parallel for
  for (uint_fast64_t i = 0; i < nelem; i++) {
    x[i] = 0.0;
    xPhys[i] = 0.0;
    dv[i] = 1.0;
  }

#pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i < gridContext.nelx + 1; i++)
    for (int k = 1; k < gridContext.nelz + 1; k++)
      for (int j = 1; j < gridContext.nely + 1; j++) {
        const int idx = i * (gridContext.wrapy - 1) * (gridContext.wrapz - 1) +
                        k * (gridContext.wrapy - 1) + j;

        x[idx] = volfrac;
        xPhys[idx] = volfrac;
      }

  applyDensityFilterGradient(gridContext, rmin, dv);

  // allocate needed memory for solver
  struct SolverData solverData;
  allocateSolverData(gridContext, nl, &solverData);
  initializeCoarseSolver(gridContext, nl - 1, &solverData.bottomSolver,
                         solverData.coarseMatrices[nl - 1]);

  int loop = 0;
  float change = 1;

#ifdef _OPENMP
  printf(" OpenMP enabled with %d threads\n", omp_get_max_threads());

  const double start_wtime = omp_get_wtime();
#endif

  /* %% START ITERATION */
  while ((change > 1e-2) && (loop < design_iters)) {

#ifdef _OPENMP
    const double loop_wtime = omp_get_wtime();
#endif

    loop++;

    int cgiter;
    float cgres;
    const int nswp = 4;
    solveMultigrid(gridContext, xPhys, nswp, nl, cgtol, &solverData, &cgiter,
                   &cgres, F, U);

    getComplianceAndSensetivity_halo(gridContext, xPhys, U, &c, dc);
    applyDensityFilterGradient(gridContext, rmin, dc);

    DTYPE vol = 0.0;
#pragma omp parallel for collapse(3) schedule(static) reduction(+ : vol)
    for (int i = 1; i < gridContext.nelx + 1; i++)
      for (int k = 1; k < gridContext.nelz + 1; k++)
        for (int j = 1; j < gridContext.nely + 1; j++) {
          const int idx =
              i * (gridContext.wrapy - 1) * (gridContext.wrapz - 1) +
              k * (gridContext.wrapy - 1) + j;

          vol += xPhys[idx];
        }

    vol /= (DTYPE)(gridContext.nelx * gridContext.nely * gridContext.nelz);
    DTYPE g = vol - volfrac;

    DTYPE l1 = 0.0, l2 = 1e9, move = 0.1;
    while ((l2 - l1) / (l1 + l2) > 1e-6) {
      DTYPE lmid = 0.5 * (l2 + l1);
      DTYPE gt = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : gt)
      for (uint_least32_t i = 0; i < nelem; i++) {
        xnew[i] =
            MAX(0.0, MAX(x[i] - move,
                         MIN(1.0, MIN(x[i] + move,
                                      x[i] * sqrt(-dc[i] / (dv[i] * lmid))))));
        gt += dv[i] * (xnew[i] - x[i]);
      }
      gt += g;
      if (gt > 0)
        l1 = lmid;
      else
        l2 = lmid;
    }

    change = 0.0;

#pragma omp parallel for schedule(static) reduction(max : change)
    for (uint_least32_t i = 0; i < nelem; i++) {
      change = MAX(change, fabs(x[i] - xnew[i]));
      x[i] = xnew[i];
    }

    applyDensityFilter(gridContext, rmin, x, xPhys);

    printf("It.:%4i Obj.:%6.3e Vol.:%6.3f ch.:%4.2f relres: %4.2e iters: %4i ",
           loop, c, vol, change, cgres, cgiter);
#ifdef _OPENMP
    printf("time: %6.3f \n", omp_get_wtime() - loop_wtime);
#else
    printf(" \n");
#endif
    fflush(NULL);
  }

#ifdef _OPENMP
  printf("End time: %6.3f \n", omp_get_wtime() - start_wtime);
#endif

  writeDensityVtkFile(nelx, nely, nelz, xPhys, "out.vtu");
  writeDensityVtkFileWithPadding(nelx, nely, nelz, xPhys, "outHalo.vtu");

  freeCoarseSolver(gridContext, nl - 1, &solverData.bottomSolver,
                   solverData.coarseMatrices[nl - 1]);
  freeSolverData(&solverData, nl);
  freeGridContext(&gridContext, nl);
}

// this function acts as a matrix-free replacement for out = (H*rho(:))./Hs
// note that rho and out cannot be the same pointer!
// temperature: cold, called once pr design iteration
void applyDensityFilter(const struct gridContext gc, const DTYPE rmin,
                        const DTYPE *rho, DTYPE *out) {

  const uint32_t nelx = gc.nelx;
  const uint32_t nely = gc.nely;
  const uint32_t nelz = gc.nelz;

  const uint32_t elWrapy = gc.wrapy - 1;
  const uint32_t elWrapz = gc.wrapz - 1;

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
#pragma omp parallel for collapse(3) schedule(static)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint64_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        out[e1] = 0.0;
        DTYPE unityScale = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const uint64_t e2 = i2 * elWrapy * elWrapz + k2 * elWrapy + j2;

              const DTYPE filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              out[e1] += filterWeight * rho[e2];
              unityScale += filterWeight;
            }
          }
        }

        out[e1] /= unityScale;
      }
}

// this function acts as a matrix-free replacement for v = H* (v(:)./Hs)
// note that rho and out cannot be the same pointer!
// temperature: cold, called twice pr design iteration
void applyDensityFilterGradient(const struct gridContext gc, const DTYPE rmin,
                                DTYPE *v) {
  const uint32_t nelx = gc.nelx;
  const uint32_t nely = gc.nely;
  const uint32_t nelz = gc.nelz;
  const uint32_t elWrapy = gc.wrapy - 1;
  const uint32_t elWrapz = gc.wrapz - 1;
  DTYPE *tmp = malloc(sizeof(DTYPE) * (gc.wrapx - 1) * elWrapy * elWrapz);

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
#pragma omp parallel for collapse(3) schedule(static)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint64_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        DTYPE unityScale = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const DTYPE filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              unityScale += filterWeight;
            }
          }
        }

        tmp[e1] = v[e1] / unityScale;
      }

// loop over elements, usually very large with nelx*nely*nelz = 100.000 or
// more
#pragma omp parallel for collapse(3) schedule(static)
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {

        const uint64_t e1 = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;

        v[e1] = 0.0;

        // loop over neighbourhood
        const uint32_t i2max = MIN(i1 + (ceil(rmin) + 1), nelx + 1);
        const uint32_t i2min = MAX(i1 - (ceil(rmin) - 1), 1);

        // the three loops herein are over a constant neighbourhood. typically
        // 4x4x4 or something like that
        for (uint32_t i2 = i2min; i2 < i2max; i2++) {

          const uint32_t k2max = MIN(k1 + (ceil(rmin) + 1), nelz + 1);
          const uint32_t k2min = MAX(k1 - (ceil(rmin) - 1), 1);

          for (uint32_t k2 = k2min; k2 < k2max; k2++) {

            const uint32_t j2max = MIN(j1 + (ceil(rmin) + 1), nely + 1);
            const uint32_t j2min = MAX(j1 - (ceil(rmin) - 1), 1);

            for (uint32_t j2 = j2min; j2 < j2max; j2++) {

              const uint64_t e2 = i2 * elWrapy * elWrapz + k2 * elWrapy + j2;

              const DTYPE filterWeight =
                  MAX(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) +
                                       (j1 - j2) * (j1 - j2) +
                                       (k1 - k2) * (k1 - k2)));

              v[e1] += filterWeight * tmp[e2];
            }
          }
        }
      }

  free(tmp);
}
