#include "write_vtk.h"

// writes a file with a snapshot of the density field (x,xPhys), can be opened
// with paraview temperature: very cold, usually called once only
void writeDensityVtkFile(const int nelx, const int nely, const int nelz,
                         const DTYPE *densityArray, const char *filename) {
  int nx = nelx + 1;
  int ny = nely + 1;
  int nz = nelz + 1;

  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int elWrapy = nely + paddingy + 3 - 1;
  const int elWrapz = nelz + paddingz + 3 - 1;

  int numberOfNodes = nx * ny * nz;
  int numberOfElements = nelx * nely * nelz;

  FILE *fid = fopen(filename, "w");

  // write header
  fprintf(fid, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
  fprintf(fid, "<UnstructuredGrid>\n");
  fprintf(fid, "<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n",
          numberOfNodes, numberOfElements);

  // points
  fprintf(fid, "<Points>\n");
  fprintf(fid,
          "<DataArray type=\"Float32\" NumberOfComponents=\"%i\" "
          "format=\"ascii\">\n",
          3);
  for (int i = 0; i < nx; i++)
    for (int k = 0; k < nz; k++)
      for (int j = 0; j < ny; j++)
        fprintf(fid, "%e %e %e\n", (float)i, (float)j, (float)k);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Points>\n");

  fprintf(fid, "<Cells>\n");

  fprintf(
      fid,
      "<DataArray type=\"Int32\" Name=\"connectivity\" format= \"ascii\">\n");
  for (int i = 0; i < nelx; i++)
    for (int k = 0; k < nelz; k++)
      for (int j = 0; j < nely; j++) {
        const int nx_1 = i;
        const int nx_2 = i + 1;
        const int nz_1 = k;
        const int nz_2 = k + 1;
        const int ny_1 = j;
        const int ny_2 = j + 1;
        fprintf(fid, "%d %d %d %d %d %d %d %d\n",
                nx_1 * ny * nz + nz_1 * ny + ny_2,
                nx_2 * ny * nz + nz_1 * ny + ny_2,
                nx_2 * ny * nz + nz_1 * ny + ny_1,
                nx_1 * ny * nz + nz_1 * ny + ny_1,
                nx_1 * ny * nz + nz_2 * ny + ny_2,
                nx_2 * ny * nz + nz_2 * ny + ny_2,
                nx_2 * ny * nz + nz_2 * ny + ny_1,
                nx_1 * ny * nz + nz_2 * ny + ny_1);
      }

  fprintf(fid, "</DataArray>\n");

  fprintf(fid,
          "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  for (int i = 1; i < numberOfElements + 1; i++)
    fprintf(fid, "%d\n", i * 8);
  fprintf(fid, "</DataArray>\n");

  fprintf(fid, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < numberOfElements; i++)
    fprintf(fid, "%d\n", 12);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Cells>\n");

  fprintf(fid, "<CellData>\n");
  fprintf(fid, "<DataArray type=\"Float32\" NumberOfComponents=\"1\" "
               "Name=\"density\" format=\"ascii\">\n");
  for (unsigned int i1 = 1; i1 < nelx + 1; i1++)
    for (unsigned int k1 = 1; k1 < nelz + 1; k1++)
      for (unsigned int j1 = 1; j1 < nely + 1; j1++) {
        const uint64_t idx = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;
        fprintf(fid, "%e\n", densityArray[idx]);
      }
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</CellData>\n");

  fprintf(fid, "</Piece>\n");
  fprintf(fid, "</UnstructuredGrid>\n");
  fprintf(fid, "</VTKFile>\n");

  fclose(fid);
}

// writes a file with a snapshot of the density field (x,xPhys), can be opened
// with paraview temperature: very cold, usually called once only
void writeDensityVtkFileWithPadding(const int nelx, const int nely,
                                    const int nelz, const DTYPE *densityArray,
                                    const char *filename) {

  const int paddingx =
      (STENCIL_SIZE_X - ((nelx + 1) % STENCIL_SIZE_X)) % STENCIL_SIZE_X;
  const int paddingy =
      (STENCIL_SIZE_Y - ((nely + 1) % STENCIL_SIZE_Y)) % STENCIL_SIZE_Y;
  const int paddingz =
      (STENCIL_SIZE_Z - ((nelz + 1) % STENCIL_SIZE_Z)) % STENCIL_SIZE_Z;

  const int wrapx = nelx + paddingx + 3;
  const int wrapy = nely + paddingy + 3;
  const int wrapz = nelz + paddingz + 3;

  const int elWrapx = wrapx - 1;
  const int elWrapy = wrapy - 1;
  const int elWrapz = wrapz - 1;

  int numberOfNodes = wrapx * wrapy * wrapz;
  int numberOfElements = elWrapx * elWrapy * elWrapz;

  FILE *fid = fopen(filename, "w");

  // write header
  fprintf(fid, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
               "byte_order=\"LittleEndian\">\n");
  fprintf(fid, "<UnstructuredGrid>\n");
  fprintf(fid, "<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n",
          numberOfNodes, numberOfElements);

  // points
  fprintf(fid, "<Points>\n");
  fprintf(fid,
          "<DataArray type=\"Float32\" NumberOfComponents=\"%i\" "
          "format=\"ascii\">\n",
          3);
  for (int i = 0; i < wrapx; i++)
    for (int k = 0; k < wrapz; k++)
      for (int j = 0; j < wrapy; j++)
        fprintf(fid, "%e %e %e\n", (float)i, (float)j, (float)k);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Points>\n");

  fprintf(fid, "<Cells>\n");

  fprintf(
      fid,
      "<DataArray type=\"Int32\" Name=\"connectivity\" format= \"ascii\">\n");
  for (int i = 0; i < elWrapx; i++)
    for (int k = 0; k < elWrapz; k++)
      for (int j = 0; j < elWrapy; j++) {
        const int nx_1 = i;
        const int nx_2 = i + 1;
        const int nz_1 = k;
        const int nz_2 = k + 1;
        const int ny_1 = j;
        const int ny_2 = j + 1;
        fprintf(fid, "%d %d %d %d %d %d %d %d\n",
                nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_1 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_1 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_2,
                nx_2 * wrapy * wrapz + nz_2 * wrapy + ny_1,
                nx_1 * wrapy * wrapz + nz_2 * wrapy + ny_1);
      }

  fprintf(fid, "</DataArray>\n");

  fprintf(fid,
          "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
  for (int i = 1; i < numberOfElements + 1; i++)
    fprintf(fid, "%d\n", i * 8);
  fprintf(fid, "</DataArray>\n");

  fprintf(fid, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < numberOfElements; i++)
    fprintf(fid, "%d\n", 12);
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</Cells>\n");

  fprintf(fid, "<CellData>\n");
  fprintf(fid, "<DataArray type=\"Float32\" NumberOfComponents=\"1\" "
               "Name=\"density\" format=\"ascii\">\n");
  for (unsigned int i1 = 0; i1 < elWrapx; i1++)
    for (unsigned int k1 = 0; k1 < elWrapz; k1++)
      for (unsigned int j1 = 0; j1 < elWrapy; j1++) {
        const uint64_t idx = i1 * elWrapy * elWrapz + k1 * elWrapy + j1;
        fprintf(fid, "%e\n", densityArray[idx]);
      }
  fprintf(fid, "</DataArray>\n");
  fprintf(fid, "</CellData>\n");

  fprintf(fid, "</Piece>\n");
  fprintf(fid, "</UnstructuredGrid>\n");
  fprintf(fid, "</VTKFile>\n");

  fclose(fid);
}
