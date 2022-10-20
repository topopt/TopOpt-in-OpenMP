#include "multigrid_solver.h"

#include "grid_utilities.h"
#include "stencil_methods.h"

// compute the norm of two vectors
// temperature: cold-medium, called 2 x number of cg iterations
CTYPE norm(CTYPE *v, uint_fast32_t size) {
  CTYPE val = 0.0;

#pragma omp parallel for reduction(+ : val) schedule(static)
  for (uint_fast32_t i = 0; i < size; i++)
    val += v[i] * v[i];
  return sqrt(val);
}

// compute the inner product of two vectors
// temperature: cold-medium, called 2 x number of cg iterations
CTYPE innerProduct(CTYPE *a, CTYPE *b, uint_fast32_t size) {
  CTYPE val = 0.0;

#pragma omp parallel for reduction(+ : val) firstprivate(size) schedule(static)
  for (uint_fast32_t i = 0; i < size; i++)
    val += a[i] * b[i];
  return val;
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.
void jacobiSmooth(const struct gridContext gc, const DTYPE *x,
                  const uint_fast32_t nswp, const CTYPE omega,
                  const MTYPE *invD, CTYPE *u, const CTYPE *b, CTYPE *tmp) {

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperator_stencil(gc, x, u, tmp);

#pragma omp parallel for collapse(3) schedule(static)
    for (int i = 1; i < gc.nelx + 2; i++)
      for (int k = 1; k < gc.nelz + 2; k++)
        for (int j = 1; j < gc.nely + 2; j++) {
          const int nidx = i * gc.wrapy * gc.wrapz + gc.wrapy * k + j;

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          u[idx1] += omega * invD[idx1] * (b[idx1] - tmp[idx1]);
          u[idx2] += omega * invD[idx2] * (b[idx2] - tmp[idx2]);
          u[idx3] += omega * invD[idx3] * (b[idx3] - tmp[idx3]);
        }
  }
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.
void jacobiSmoothCoarse(const struct gridContext gc, const DTYPE *x,
                        const int l, const uint_fast32_t nswp,
                        const CTYPE omega, const MTYPE *invD, CTYPE *u,
                        const CTYPE *b, CTYPE *tmp) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspace_halo(gc, l, x, u, tmp);

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp parallel for collapse(3) schedule(static)
    for (int i = 1; i < nelxc + 2; i++)
      for (int k = 1; k < nelzc + 2; k++)
        for (int j = 1; j < nelyc + 2; j++) {
          const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          u[idx1] += omega * invD[idx1] * (b[idx1] - tmp[idx1]);
          u[idx2] += omega * invD[idx2] * (b[idx2] - tmp[idx2]);
          u[idx3] += omega * invD[idx3] * (b[idx3] - tmp[idx3]);
        }
  }
}

// jacobi smoothing/preconditioning
// temperature: hot, called 2x(number of levels)x(number of cg iterations) ~
// [20-1000] times every design iteration. Note that most compute time is spent
// in child function.
void jacobiSmoothCoarseAssembled(const struct gridContext gc,
                                 const struct CSRMatrix M, const int l,
                                 const uint_fast32_t nswp, const CTYPE omega,
                                 const MTYPE *invD, CTYPE *u, const CTYPE *b,
                                 CTYPE *tmp) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;

  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;

  // usually nswp is between 1 and 5
  for (int s = 0; s < nswp; s++) {
    applyStateOperatorSubspaceMatrix(gc, l, M, u, tmp);

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel.
#pragma omp parallel for collapse(3) schedule(static)
    for (int i = 1; i < nelxc + 2; i++)
      for (int k = 1; k < nelzc + 2; k++)
        for (int j = 1; j < nelyc + 2; j++) {
          const int nidx = (i * wrapyc * wrapzc + wrapyc * k + j);

          const uint32_t idx1 = 3 * nidx + 0;
          const uint32_t idx2 = 3 * nidx + 1;
          const uint32_t idx3 = 3 * nidx + 2;

          u[idx1] += omega * invD[idx1] * (b[idx1] - tmp[idx1]);
          u[idx2] += omega * invD[idx2] * (b[idx2] - tmp[idx2]);
          u[idx3] += omega * invD[idx3] * (b[idx3] - tmp[idx3]);
        }
  }
}

// Vcycle preconditioner. recursive function.
// temperature: medium, called (number of levels)x(number of cg iterations ~
// 5 - 100) every design iteration. Much of the compute time is spent in
// this function, although in children functions.
void VcyclePreconditioner(const struct gridContext gc, const DTYPE *x,
                          const int nl, const int l, MTYPE **const invD,
                          struct CoarseSolverData *bottomSolverData,
                          const struct CSRMatrix *coarseMatrices, CTYPE omega,
                          const int nswp, CTYPE **r, CTYPE **z, CTYPE **d) {

  const int ncell = pow(2, l);
  const int32_t nelxc = gc.nelx / ncell;
  const int32_t nelyc = gc.nely / ncell;
  const int32_t nelzc = gc.nelz / ncell;
  const int paddingxc =
      (STENCIL_SIZE_X - ((nelxc + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingyc =
      (STENCIL_SIZE_Y - ((nelyc + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingzc =
      (STENCIL_SIZE_Z - ((nelzc + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;
  const int wrapxc = nelxc + paddingxc + 3;
  const int wrapyc = nelyc + paddingyc + 3;
  const int wrapzc = nelzc + paddingzc + 3;
  const uint_fast32_t ndofc = 3 * wrapyc * wrapzc * wrapxc;

  CTYPE *zptr = z[l];
  CTYPE *dptr = d[l];
  CTYPE *rptr = r[l];
  CTYPE *next_rptr = r[l + 1];
  CTYPE *next_zptr = z[l + 1];
  MTYPE *invDptr = invD[l];

// zero z[l]
// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel
#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndofc; i++)
    zptr[i] = 0.0;

  // smooth
  if (l == 0) {

    jacobiSmooth(gc, x, nswp, omega, invDptr, zptr, rptr, dptr);

    applyStateOperator_stencil(gc, x, zptr, dptr);

  } else if (l < number_of_matrix_free_levels) {
    jacobiSmoothCoarse(gc, x, l, nswp, omega, invDptr, zptr, rptr, dptr);
    applyStateOperatorSubspace_halo(gc, l, x, zptr, dptr);
  } else {
    jacobiSmoothCoarseAssembled(gc, coarseMatrices[l], l, nswp, omega, invDptr,
                                zptr, rptr, dptr);
    applyStateOperatorSubspaceMatrix(gc, l, coarseMatrices[l], zptr, dptr);
  }

// long for loop, as ndof is typically 300.000 or more, but also trivially
// parallel
#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndofc; i++)
    dptr[i] = rptr[i] - dptr[i];

  // project residual down
  projectToCoarserGrid_halo(gc, l, dptr, next_rptr);

  // smooth coarse
  if (nl == l + 2) {
    const int ncell_nl = pow(2, l + 1);
    const int32_t nelx_nl = gc.nelx / ncell_nl;
    const int32_t nely_nl = gc.nely / ncell_nl;
    const int32_t nelz_nl = gc.nelz / ncell_nl;
    const int paddingx_nl =
        (STENCIL_SIZE_X - ((nelx_nl + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
    const int paddingy_nl =
        (STENCIL_SIZE_Y - ((nely_nl + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
    const int paddingz_nl =
        (STENCIL_SIZE_Z - ((nelz_nl + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;
    const int wrapx_nl = nelx_nl + paddingx_nl + 3;
    const int wrapy_nl = nely_nl + paddingy_nl + 3;
    const int wrapz_nl = nelz_nl + paddingz_nl + 3;
    const uint_fast32_t ndof_nl = 3 * wrapy_nl * wrapz_nl * wrapx_nl;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < ndof_nl; i++)
      next_zptr[i] = 0.0;

    solveSubspaceMatrix(gc, l + 1, *bottomSolverData, next_rptr, next_zptr);

  } else
    VcyclePreconditioner(gc, x, nl, l + 1, invD, bottomSolverData,
                         coarseMatrices, omega, nswp, r, z, d);

  // project residual up
  projectToFinerGrid_halo(gc, l, z[l + 1], d[l]);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < ndofc; i++)
    zptr[i] += dptr[i];

  // smooth
  if (l == 0) {
    jacobiSmooth(gc, x, nswp, omega, invDptr, zptr, rptr, dptr);

  } else if (l < number_of_matrix_free_levels) {

    jacobiSmoothCoarse(gc, x, l, nswp, omega, invDptr, zptr, rptr, dptr);
  } else {

    jacobiSmoothCoarseAssembled(gc, coarseMatrices[l], l, nswp, omega, invDptr,
                                zptr, rptr, dptr);
  }
}

// solves the linear system of Ku = b.
// temperature: medium, accounts for 95% or more of runtime, but this time is
// spent in children functions. The iter loop of this funciton is a good
// candidate for GPU parallel region scope, as it is only performed once every
// design iteration (and thus only 100 times during a program)
void solveMultigrid(const struct gridContext gc, DTYPE *x, const int nswp,
                    const int nl, const CTYPE tol, struct SolverData *data,
                    int *finalIter, float *finalRes, CTYPE *b, STYPE *u) {

  const uint_fast32_t ndof = 3 * gc.wrapx * gc.wrapy * gc.wrapz;

  CTYPE *r = data->r;
  CTYPE *p = data->p;
  CTYPE *q = data->q;
  CTYPE *z = data->z;

  MTYPE **invD = data->invD;
  CTYPE **dmg = data->dmg;
  CTYPE **rmg = data->rmg;
  CTYPE **zmg = data->zmg;

  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    // printf("assemble mat l:%i\n", l);
    assembleSubspaceMatrix(gc, l, x, data->coarseMatrices[l], invD[l]);
  }

  for (int l = 0; l < nl; l++) {
    assembleInvertedMatrixDiagonalSubspace_halo(gc, x, l, invD[l]);
  }

  factorizeSubspaceMatrix(gc, nl - 1, data->bottomSolver,
                          data->coarseMatrices[nl - 1]);

  CTYPE rhoold = 0.0;
  CTYPE dpr;
  CTYPE alpha;
  CTYPE rho;

  // setup residual vector
#pragma omp parallel for
  for (uint_fast32_t i = 0; i < ndof; i++)
    z[i] = (CTYPE)u[i];

  applyStateOperator_stencil(gc, x, z, r);

#pragma omp parallel for
  for (uint_fast32_t i = 0; i < ndof; i++)
    r[i] = b[i] - r[i];

  // setup scalars
  const MTYPE omega = 0.6;
  const CTYPE bnorm = norm(b, ndof);
  const int maxIter = 1000;

  // begin cg loop - usually spans 5 - 300 iterations will be reduced to 5 -
  // 20 iterations once direct solver is included for coarse subproblem.
  for (int iter = 0; iter < maxIter; iter++) {

    // get preconditioned vector
    VcyclePreconditioner(gc, x, nl, 0, invD, &data->bottomSolver,
                         data->coarseMatrices, omega, nswp, rmg, zmg, dmg);

    rho = innerProduct(r, z, ndof);

    if (iter == 0) {

#pragma omp parallel for
      for (uint_fast32_t i = 0; i < ndof; i++)
        p[i] = z[i];

    } else {

      CTYPE beta = rho / rhoold;
#pragma omp parallel for firstprivate(beta)
      for (uint_fast32_t i = 0; i < ndof; i++)
        p[i] = beta * p[i] + z[i];
    }

    applyStateOperator_stencil(gc, x, p, q);

    dpr = innerProduct(p, q, ndof);
    alpha = rho / dpr;
    rhoold = rho;

#pragma omp parallel for firstprivate(alpha)
    for (uint_fast32_t i = 0; i < ndof; i++)
      u[i] += (STYPE)(alpha * p[i]);

#pragma omp parallel for firstprivate(alpha)
    for (uint_fast32_t i = 0; i < ndof; i++)
      r[i] -= alpha * q[i];

    const CTYPE rnorm = norm(r, ndof);
    const CTYPE relres = rnorm / bnorm;

    (*finalIter) = iter;
    (*finalRes) = relres;

    // printf("it: %i, res=%e\n", iter, relres);

    if (relres < tol)
      break;
  }
}

void allocateSolverData(const struct gridContext gc, const int nl,
                        struct SolverData *data) {

  allocateStateField(gc, 0, &(*data).r);
  allocateStateField(gc, 0, &(*data).p);
  allocateStateField(gc, 0, &(*data).q);
  allocateStateField(gc, 0, &(*data).z);

  (*data).invD = malloc(sizeof(MTYPE *) * nl);
  (*data).dmg = malloc(sizeof(CTYPE *) * nl);
  (*data).rmg = malloc(sizeof(CTYPE *) * nl);
  (*data).zmg = malloc(sizeof(CTYPE *) * nl);

  allocateStateField(gc, 0, &((*data).dmg[0]));
  allocateStateField_MTYPE(gc, 0, &((*data).invD[0]));
  (*data).rmg[0] = (*data).r;
  (*data).zmg[0] = (*data).z;

  for (int l = 1; l < nl; l++) {
    allocateStateField(gc, l, &((*data).dmg[l]));
    allocateStateField(gc, l, &((*data).rmg[l]));
    allocateStateField(gc, l, &((*data).zmg[l]));
    allocateStateField_MTYPE(gc, l, &((*data).invD[l]));
  }

  // allocate for all levels for easy indces
  (*data).coarseMatrices = malloc(sizeof(struct CSRMatrix) * nl);
  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    allocateSubspaceMatrix(gc, l, &((*data).coarseMatrices[l]));
  }
}

void freeSolverData(struct SolverData *data, const int nl) {

  free((*data).r);
  free((*data).z);
  free((*data).p);
  free((*data).q);

  free((*data).invD[0]);
  free((*data).dmg[0]);

  for (int l = 1; l < nl; l++) {
    free((*data).invD[l]);
    free((*data).dmg[l]);
    free((*data).rmg[l]);
    free((*data).zmg[l]);
  }

  free((*data).invD);
  free((*data).dmg);
  free((*data).zmg);
  free((*data).rmg);

  for (int l = number_of_matrix_free_levels; l < nl; l++) {
    freeSubspaceMatrix(&((*data).coarseMatrices[l]));
  }
  free((*data).coarseMatrices);
}
