#pragma once

#include "definitions.h"

// compute the contitutive matrix
// temperature: frozen, called only in preprocessing
void getC(MTYPE C[6][6],   /* out */
          const MTYPE nu); /*  in */

// compute the strain-displacement matrix
// temperature: frozen, called only in preprocessing
void getB(MTYPE B[6][24],      /* out */
          MTYPE *jdet,         /* out */
          const MTYPE iso[3],  /*  in */
          const MTYPE xe[24]); /*  in */

// compute the local stiffness matrix
// temperature: frozen, called only in preprocessing
void getKEsubspace(MTYPE *KEarray, /* out */
                   const MTYPE nu, /* in */
                   const int l);   /* in */
