#include "local_matrix.h"

#include <cblas.h>

// compute the contitutive matrix
// temperature: frozen, called only in preprocessing
void getC(MTYPE C[6][6],  /* out */
          const MTYPE nu) /*  in */
{
  const MTYPE temp1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const MTYPE temp2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const MTYPE temp3 = 1.0 / (2.0 * (1.0 + nu));

  for (unsigned int i = 0; i < 6 * 6; i++)
    *((MTYPE *)C + i) = 0.0;

  C[0][0] = temp1;
  C[1][1] = temp1;
  C[2][2] = temp1;
  C[3][3] = temp3;
  C[4][4] = temp3;
  C[5][5] = temp3;
  C[0][1] = temp2;
  C[1][0] = temp2;
  C[0][2] = temp2;
  C[2][0] = temp2;
  C[1][2] = temp2;
  C[2][1] = temp2;
}

// compute the strain-displacement matrix
// temperature: frozen, called only in preprocessing
void getB(MTYPE B[6][24],     /* out */
          MTYPE *jdet,        /* out */
          const MTYPE iso[3], /*  in */
          const MTYPE xe[24]) /*  in */
{
  /*     xi = iso(1); */
  const MTYPE xi = iso[1 - 1];
  /*     eta = iso(2); */
  const MTYPE eta = iso[2 - 1];
  /*     zeta = iso(3); */
  const MTYPE zeta = iso[3 - 1];

  const MTYPE n1xi = -0.125 * (1 - eta) * (1 - zeta);
  const MTYPE n1eta = -0.125 * (1 - xi) * (1 - zeta);
  const MTYPE n1zeta = -0.125 * (1 - xi) * (1 - eta);
  const MTYPE n2xi = 0.125 * (1 - eta) * (1 - zeta);
  const MTYPE n2eta = -0.125 * (1 + xi) * (1 - zeta);
  const MTYPE n2zeta = -0.125 * (1 + xi) * (1 - eta);

  const MTYPE n3xi = 0.125 * (1 + eta) * (1 - zeta);
  const MTYPE n3eta = 0.125 * (1 + xi) * (1 - zeta);
  const MTYPE n3zeta = -0.125 * (1 + xi) * (1 + eta);
  const MTYPE n4xi = -0.125 * (1 + eta) * (1 - zeta);
  const MTYPE n4eta = 0.125 * (1 - xi) * (1 - zeta);
  const MTYPE n4zeta = -0.125 * (1 - xi) * (1 + eta);

  const MTYPE n5xi = -0.125 * (1 - eta) * (1 + zeta);
  const MTYPE n5eta = -0.125 * (1 - xi) * (1 + zeta);
  const MTYPE n5zeta = 0.125 * (1 - xi) * (1 - eta);
  const MTYPE n6xi = 0.125 * (1 - eta) * (1 + zeta);
  const MTYPE n6eta = -0.125 * (1 + xi) * (1 + zeta);
  const MTYPE n6zeta = 0.125 * (1 + xi) * (1 - eta);

  const MTYPE n7xi = 0.125 * (1 + eta) * (1 + zeta);
  const MTYPE n7eta = 0.125 * (1 + xi) * (1 + zeta);
  const MTYPE n7zeta = 0.125 * (1 + xi) * (1 + eta);
  const MTYPE n8xi = -0.125 * (1 + eta) * (1 + zeta);
  const MTYPE n8eta = 0.125 * (1 - xi) * (1 + zeta);
  const MTYPE n8zeta = 0.125 * (1 - xi) * (1 + eta);

  MTYPE L[6][9];
  for (unsigned int i = 0; i < 6 * 9; i++)
    *((MTYPE *)L + i) = 0.0;

  MTYPE jac[3][3];
  for (unsigned int i = 0; i < 3 * 3; i++)
    *((MTYPE *)jac + i) = 0.0;

  MTYPE jacinvt[9][9];
  for (unsigned int i = 0; i < 9 * 9; i++)
    *((MTYPE *)jacinvt + i) = 0.0;

  MTYPE Nt[9][24];
  for (unsigned int i = 0; i < 9 * 24; i++)
    *((MTYPE *)Nt + i) = 0.0;

  L[0][0] = 1.0;
  L[1][4] = 1.0;
  L[2][8] = 1.0;
  L[3][1] = 1.0;
  L[3][3] = 1.0;
  L[4][5] = 1.0;
  L[4][7] = 1.0;
  L[5][2] = 1.0;
  L[5][6] = 1.0;

  Nt[0][0] = n1xi;
  Nt[1][0] = n1eta;
  Nt[2][0] = n1zeta;
  Nt[0][3] = n2xi;
  Nt[1][3] = n2eta;
  Nt[2][3] = n2zeta;
  Nt[0][6] = n3xi;
  Nt[1][6] = n3eta;
  Nt[2][6] = n3zeta;
  Nt[0][9] = n4xi;
  Nt[1][9] = n4eta;
  Nt[2][9] = n4zeta;
  Nt[0][12] = n5xi;
  Nt[1][12] = n5eta;
  Nt[2][12] = n5zeta;
  Nt[0][15] = n6xi;
  Nt[1][15] = n6eta;
  Nt[2][15] = n6zeta;
  Nt[0][18] = n7xi;
  Nt[1][18] = n7eta;
  Nt[2][18] = n7zeta;
  Nt[0][21] = n8xi;
  Nt[1][21] = n8eta;
  Nt[2][21] = n8zeta;

  Nt[3][1] = n1xi;
  Nt[4][1] = n1eta;
  Nt[5][1] = n1zeta;
  Nt[3][4] = n2xi;
  Nt[4][4] = n2eta;
  Nt[5][4] = n2zeta;
  Nt[3][7] = n3xi;
  Nt[4][7] = n3eta;
  Nt[5][7] = n3zeta;
  Nt[3][10] = n4xi;
  Nt[4][10] = n4eta;
  Nt[5][10] = n4zeta;
  Nt[3][13] = n5xi;
  Nt[4][13] = n5eta;
  Nt[5][13] = n5zeta;
  Nt[3][16] = n6xi;
  Nt[4][16] = n6eta;
  Nt[5][16] = n6zeta;
  Nt[3][19] = n7xi;
  Nt[4][19] = n7eta;
  Nt[5][19] = n7zeta;
  Nt[3][22] = n8xi;
  Nt[4][22] = n8eta;
  Nt[5][22] = n8zeta;

  Nt[6][2] = n1xi;
  Nt[7][2] = n1eta;
  Nt[8][2] = n1zeta;
  Nt[6][5] = n2xi;
  Nt[7][5] = n2eta;
  Nt[8][5] = n2zeta;
  Nt[6][8] = n3xi;
  Nt[7][8] = n3eta;
  Nt[8][8] = n3zeta;
  Nt[6][11] = n4xi;
  Nt[7][11] = n4eta;
  Nt[8][11] = n4zeta;
  Nt[6][14] = n5xi;
  Nt[7][14] = n5eta;
  Nt[8][14] = n5zeta;
  Nt[6][17] = n6xi;
  Nt[7][17] = n6eta;
  Nt[8][17] = n6zeta;
  Nt[6][20] = n7xi;
  Nt[7][20] = n7eta;
  Nt[8][20] = n7zeta;
  Nt[6][23] = n8xi;
  Nt[7][23] = n8eta;
  Nt[8][23] = n8zeta;

  jac[0][0] = n1xi * xe[0] + n2xi * xe[3] + n3xi * xe[6] + n4xi * xe[9] +
              n5xi * xe[12] + n6xi * xe[15] + n7xi * xe[18] + n8xi * xe[21];

  jac[1][0] = n1eta * xe[0] + n2eta * xe[3] + n3eta * xe[6] + n4eta * xe[9] +
              n5eta * xe[12] + n6eta * xe[15] + n7eta * xe[18] + n8eta * xe[21];

  jac[2][0] = n1zeta * xe[0] + n2zeta * xe[3] + n3zeta * xe[6] +
              n4zeta * xe[9] + n5zeta * xe[12] + n6zeta * xe[15] +
              n7zeta * xe[18] + n8zeta * xe[21];

  jac[0][1] = n1xi * xe[1] + n2xi * xe[4] + n3xi * xe[7] + n4xi * xe[10] +
              n5xi * xe[13] + n6xi * xe[16] + n7xi * xe[19] + n8xi * xe[22];

  jac[1][1] = n1eta * xe[1] + n2eta * xe[4] + n3eta * xe[7] + n4eta * xe[10] +
              n5eta * xe[13] + n6eta * xe[16] + n7eta * xe[19] + n8eta * xe[22];

  jac[2][1] = n1zeta * xe[1] + n2zeta * xe[4] + n3zeta * xe[7] +
              n4zeta * xe[10] + n5zeta * xe[13] + n6zeta * xe[16] +
              n7zeta * xe[19] + n8zeta * xe[22];

  jac[0][2] = n1xi * xe[2] + n2xi * xe[5] + n3xi * xe[8] + n4xi * xe[11] +
              n5xi * xe[14] + n6xi * xe[17] + n7xi * xe[20] + n8xi * xe[23];

  jac[1][2] = n1eta * xe[2] + n2eta * xe[5] + n3eta * xe[8] + n4eta * xe[11] +
              n5eta * xe[14] + n6eta * xe[17] + n7eta * xe[20] + n8eta * xe[23];

  jac[2][2] = n1zeta * xe[2] + n2zeta * xe[5] + n3zeta * xe[8] +
              n4zeta * xe[11] + n5zeta * xe[14] + n6zeta * xe[17] +
              n7zeta * xe[20] + n8zeta * xe[23];

  /*     jdet = det(jac); */
  //  det(A)=A11(A22A33−A23A32)−A12(A21A33−A23A31)+A13(A21A32−A22A31)
  (*jdet) = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[1][2]) -
            jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0]) +
            jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);

  /*     ijac = inv(jac); */
  // https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
  jacinvt[0][0] =
      (1.0 / (*jdet)) * (jac[1][1] * jac[2][2] - jac[1][2] * jac[1][2]);
  jacinvt[0][1] =
      (1.0 / (*jdet)) * -1.0 * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0]);
  jacinvt[0][2] =
      (1.0 / (*jdet)) * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);

  jacinvt[1][0] =
      (1.0 / (*jdet)) * -1.0 * (jac[0][1] * jac[2][2] - jac[0][2] * jac[2][1]);
  jacinvt[1][1] =
      (1.0 / (*jdet)) * (jac[0][0] * jac[2][2] - jac[0][2] * jac[2][0]);
  jacinvt[1][2] =
      (1.0 / (*jdet)) * -1.0 * (jac[0][0] * jac[2][1] - jac[0][1] * jac[2][0]);

  jacinvt[2][0] =
      (1.0 / (*jdet)) * (jac[0][1] * jac[1][2] - jac[0][2] * jac[1][1]);
  jacinvt[2][1] =
      (1.0 / (*jdet)) * -1.0 * (jac[0][0] * jac[1][2] - jac[0][2] * jac[1][0]);
  jacinvt[2][2] =
      (1.0 / (*jdet)) * (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]);

  for (int k = 1; k < 3; k++)
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        jacinvt[3 * k + i][3 * k + j] = jacinvt[i][j];

  MTYPE partial[9][24];

  /*! \todo add checks so that double / single precision GEMM
    is selected. */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 9, 24, 9, 1.0,
              (MTYPE *)jacinvt, 9, (MTYPE *)Nt, 24, 0.0, (MTYPE *)partial, 24);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 6, 24, 9, 1.0,
              (MTYPE *)L, 9, (MTYPE *)partial, 24, 0.0, (MTYPE *)B, 24);
}

// compute the local stiffness matrix
// temperature: frozen, called only in preprocessing
void getKEsubspace(MTYPE *KEarray, /* out */
                   const MTYPE nu, /* in */
                   const int l) {  /* in */

  const int ncell = pow(2, l);
  const int integrationPoints = 25;

  const MTYPE a = 0.5 * ncell;
  const MTYPE b = 0.5 * ncell;
  const MTYPE c = 0.5 * ncell;

  const MTYPE xe[24] = {-a, -b, -c, a, -b, -c, a, b, -c, -a, b, -c,
                        -a, -b, c,  a, -b, c,  a, b, c,  -a, b, c};

  MTYPE C[6][6];
  getC(C, nu);

  const MTYPE spacing = 2.0 / (MTYPE)ncell / (MTYPE)integrationPoints;
  const MTYPE subCellVolume = spacing * spacing * spacing;

  MTYPE iso[3];
  MTYPE B[6][24];
  MTYPE partial[24][6];
  MTYPE jdet;

  for (int i = 0; i < 24 * 24 * ncell * ncell * ncell; i++)
    KEarray[i] = 0.0;

  for (int ii = 0; ii < ncell; ii++)
    for (int kk = 0; kk < ncell; kk++)
      for (int jj = 0; jj < ncell; jj++) {

        const int cellidx = ncell * ncell * ii + ncell * kk + jj;

        MTYPE igp = -1.0 + spacing / 2.0 + 2.0 / (MTYPE)ncell * ((MTYPE)ii);

        for (int iii = 0; iii < integrationPoints; iii++) {

          MTYPE jgp = -1.0 + spacing / 2.0 + 2.0 / (MTYPE)ncell * ((MTYPE)jj);

          for (int jjj = 0; jjj < integrationPoints; jjj++) {

            MTYPE kgp = -1.0 + spacing / 2.0 + 2.0 / (MTYPE)ncell * ((MTYPE)kk);

            for (int kkk = 0; kkk < integrationPoints; kkk++) {

              iso[0] = igp;
              iso[1] = -1.0 * jgp; /*important to flip y/eta-coordinate, due to
                                  bad numbering in original code*/
              iso[2] = kgp;

              getB(B, &jdet, iso, xe);

              cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 24, 6, 6,
                          1.0, (MTYPE *)B, 24, (MTYPE *)C, 6, 0.0,
                          (MTYPE *)partial, 6);

              cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 24, 24, 6,
                          jdet * subCellVolume, (MTYPE *)partial, 6, (MTYPE *)B,
                          24, 1.0, KEarray + 24 * 24 * cellidx, 24);

              kgp += spacing;
            }
            jgp += spacing;
          }
          igp += spacing;
        }
      }
}
