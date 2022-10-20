extern "C" {
#include "grid_utilities.h"
#include "local_matrix.h"
#include "stencil_methods.h"
#include "stencil_optimization.h"
}

#include <benchmark/benchmark.h>

static void applyOperator_stencil(benchmark::State &state) {

  struct gridContext gc;
  gc.E0 = 1;
  gc.Emin = 1e-6;
  gc.nu = 0.3;
  gc.elementSizeX = 0.5;
  gc.elementSizeY = 0.5;
  gc.elementSizeZ = 0.5;
  gc.nelx = state.range(0);
  gc.nely = state.range(0) / 2;
  gc.nelz = state.range(0) / 2;
  gc.penal = 3;

  initializeGridContext(&gc, 1);

  const int ne_stencil = (gc.wrapx - 1) * (gc.wrapy - 1) * (gc.wrapz - 1);

  DTYPE *rho_s = (DTYPE *)malloc(sizeof(DTYPE) * ne_stencil);

  for (int i = 0; i < ne_stencil; i++) {
    rho_s[i] = 0.0;
  }

  for (uint i = 1; i < gc.nelx + 1; i++)
    for (uint k = 1; k < gc.nelz + 1; k++)
      for (uint j = 1; j < gc.nely + 1; j++) {
        rho_s[(i * (gc.wrapy - 1) * (gc.wrapz - 1) + (gc.wrapy - 1) * k + j)] =
            1.0;
      }

  CTYPE *r_s;
  CTYPE *u_s;
  allocateStateField(gc, 0, &u_s);
  allocateStateField(gc, 0, &r_s);

  r_s[3 * ((gc.nelx + 1) * gc.wrapy * gc.wrapz + gc.wrapy + 1) + 1] = 100.0;

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    applyStateOperator_stencil(gc, rho_s, r_s, u_s);
    benchmark::DoNotOptimize(u_s);
    benchmark::ClobberMemory();
  }

  const int ndof = 3 * (gc.nely + 1) * (gc.nelx + 1) * (gc.nelz + 1);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ndof);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * ndof * 8);

  freeGridContext(&gc, 1);
  free(rho_s);
  free(r_s);
  free(u_s);
}

static void applyOperatorSubspace_halo(benchmark::State &state) {

  struct gridContext gc;
  gc.E0 = 1;
  gc.Emin = 1e-6;
  gc.nu = 0.3;
  gc.elementSizeX = 0.5;
  gc.elementSizeY = 0.5;
  gc.elementSizeZ = 0.5;
  gc.nelx = state.range(0);
  gc.nely = state.range(0) / 2;
  gc.nelz = state.range(0) / 2;
  gc.penal = 3;

  const int l = 2;

  initializeGridContext(&gc, l + 1);

  const int ne_stencil = (gc.wrapx - 1) * (gc.wrapy - 1) * (gc.wrapz - 1);

  DTYPE *rho_s = (DTYPE *)malloc(sizeof(DTYPE) * ne_stencil);

  for (int i = 0; i < ne_stencil; i++) {
    rho_s[i] = 0.0;
  }

  for (uint i = 1; i < gc.nelx + 1; i++)
    for (uint k = 1; k < gc.nelz + 1; k++)
      for (uint j = 1; j < gc.nely + 1; j++) {
        rho_s[(i * (gc.wrapy - 1) * (gc.wrapz - 1) + (gc.wrapy - 1) * k + j)] =
            1.0;
      }

  CTYPE *r_s;
  CTYPE *u_s;
  allocateStateField(gc, l, &u_s);
  allocateStateField(gc, l, &r_s);

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    applyStateOperatorSubspace_halo(gc, l, rho_s, r_s, u_s);
    benchmark::DoNotOptimize(u_s);
    benchmark::ClobberMemory();
  }

  const int ndof =
      3 * (gc.nely / 2 + 1) * (gc.nelx / 2 + 1) * (gc.nelz / 2 + 1);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ndof);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * ndof * 8);

  freeGridContext(&gc, l + 1);
  free(rho_s);
  free(r_s);
  free(u_s);
}

static void projectToCoarse_stencil(benchmark::State &state) {

  struct gridContext gc;
  gc.E0 = 1;
  gc.Emin = 1e-6;
  gc.nu = 0.3;
  gc.elementSizeX = 0.5;
  gc.elementSizeY = 0.5;
  gc.elementSizeZ = 0.5;
  gc.nelx = state.range(0);
  gc.nely = state.range(0) / 2;
  gc.nelz = state.range(0) / 2;
  gc.penal = 3;

  initializeGridContext(&gc, 1);

  const int ne_stencil = (gc.wrapx - 1) * (gc.wrapy - 1) * (gc.wrapz - 1);

  DTYPE *rho_s = (DTYPE *)malloc(sizeof(DTYPE) * ne_stencil);

  for (int i = 0; i < ne_stencil; i++) {
    rho_s[i] = 0.0;
  }

  for (uint i = 1; i < gc.nelx + 1; i++)
    for (uint k = 1; k < gc.nelz + 1; k++)
      for (uint j = 1; j < gc.nely + 1; j++) {
        rho_s[(i * (gc.wrapy - 1) * (gc.wrapz - 1) + (gc.wrapy - 1) * k + j)] =
            1.0;
      }

  CTYPE *u;
  CTYPE *u_c;
  allocateStateField(gc, 0, &u);
  allocateStateField(gc, 1, &u_c);

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    projectToCoarserGrid_halo(gc, 0, u, u_c);
    benchmark::DoNotOptimize(u_c);
    benchmark::ClobberMemory();
  }

  const int ndof = 3 * (gc.nely + 1) * (gc.nelx + 1) * (gc.nelz + 1);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ndof);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * ndof * 8);

  freeGridContext(&gc, 1);
  free(rho_s);
  free(u);
  free(u_c);
}

static void projectToFine_stencil(benchmark::State &state) {

  struct gridContext gc;
  gc.E0 = 1;
  gc.Emin = 1e-6;
  gc.nu = 0.3;
  gc.elementSizeX = 0.5;
  gc.elementSizeY = 0.5;
  gc.elementSizeZ = 0.5;
  gc.nelx = state.range(0);
  gc.nely = state.range(0) / 2;
  gc.nelz = state.range(0) / 2;
  gc.penal = 3;

  initializeGridContext(&gc, 1);

  const int ne_stencil = (gc.wrapx - 1) * (gc.wrapy - 1) * (gc.wrapz - 1);

  DTYPE *rho_s = (DTYPE *)malloc(sizeof(DTYPE) * ne_stencil);

  for (int i = 0; i < ne_stencil; i++) {
    rho_s[i] = 0.0;
  }

  for (uint i = 1; i < gc.nelx + 1; i++)
    for (uint k = 1; k < gc.nelz + 1; k++)
      for (uint j = 1; j < gc.nely + 1; j++) {
        rho_s[(i * (gc.wrapy - 1) * (gc.wrapz - 1) + (gc.wrapy - 1) * k + j)] =
            1.0;
      }

  CTYPE *u;
  CTYPE *u_c;
  allocateStateField(gc, 0, &u);
  allocateStateField(gc, 1, &u_c);

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    projectToFinerGrid_halo(gc, 0, u_c, u);
    benchmark::DoNotOptimize(u_c);
    benchmark::ClobberMemory();
  }

  const int ndof = 3 * (gc.nely + 1) * (gc.nelx + 1) * (gc.nelz + 1);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * ndof);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * ndof * 8);

  freeGridContext(&gc, 1);
  free(rho_s);
  free(u);
  free(u_c);
}

static void densityFilter(benchmark::State &state) {

  struct gridContext gc;
  gc.E0 = 1;
  gc.Emin = 1e-6;
  gc.nu = 0.3;
  gc.elementSizeX = 0.5;
  gc.elementSizeY = 0.5;
  gc.elementSizeZ = 0.5;
  gc.nelx = state.range(0);
  gc.nely = state.range(0) / 2;
  gc.nelz = state.range(0) / 2;
  gc.penal = 3;

  const int paddingx =
      (STENCIL_SIZE_X - ((gc.nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((gc.nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((gc.nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  gc.wrapx = gc.nelx + paddingx + 3;
  gc.wrapy = gc.nely + paddingy + 3;
  gc.wrapz = gc.nelz + paddingz + 3;

  const DTYPE rmin = 2.5;

  const int ne_stencil = (gc.wrapx - 1) * (gc.wrapy - 1) * (gc.wrapz - 1);

  DTYPE *rho_s = (DTYPE *)malloc(sizeof(DTYPE) * ne_stencil);
  DTYPE *out_s = (DTYPE *)malloc(sizeof(DTYPE) * ne_stencil);

  for (int i = 0; i < ne_stencil; i++) {
    rho_s[i] = 0.0;
  }

  for (int i = 0; i < ne_stencil; i++) {
    out_s[i] = 0.0;
  }

  for (uint i = 1; i < gc.nelx + 1; i++)
    for (uint k = 1; k < gc.nelz + 1; k++)
      for (uint j = 1; j < gc.nely + 1; j++) {
        rho_s[(i * (gc.wrapy - 1) * (gc.wrapz - 1) + (gc.wrapy - 1) * k + j)] =
            0.5;
      }

  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    applyDensityFilter(gc, rmin, rho_s, out_s);
    benchmark::DoNotOptimize(out_s);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          ne_stencil);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          ne_stencil * 4);

  free(rho_s);
}

// Register benchmarks
BENCHMARK(densityFilter)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(applyOperator_stencil)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(applyOperatorSubspace_halo)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(projectToCoarse_stencil)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(projectToFine_stencil)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 3)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
// Run the benchmark
BENCHMARK_MAIN();