/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  peano.h
 *
 *  \brief declaration of function prototypes used for Peano-Hilbert keys
 */

#ifndef SORT_H
#define SORT_H

#include "gadget/dtypes.h"

class peanokey
{
 public:
  MyIntPosType hs, is, ls; /* 'hs'-high significance, 'is'-intermediate, 'ls'-low significance bits */
};

bool operator>=(const peanokey &a, const peanokey &b);

bool operator<(const peanokey &a, const peanokey &b);

peanokey operator+(const peanokey &a, const peanokey &b);

peanokey get_peanokey_offset(unsigned int j, int bits); /* this returns the peanokey for which  j << bits */

peanokey peano_hilbert_key(MyIntPosType x, MyIntPosType y, MyIntPosType z, int bits);
void peano_hilbert_key_inverse(peanokey key, int bits, MyIntPosType *x, MyIntPosType *y, MyIntPosType *z);

unsigned char peano_incremental_key(unsigned char pix, unsigned char *rotation);

#endif
