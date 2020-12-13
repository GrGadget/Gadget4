/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  peano.cc
 *
 *  \brief routines for computing Peano-Hilbert keys and for bringing particles into this order
 */

#include "gadgetconfig.h"

//#include "../sort/cxxsort.h"
#include "../sort/peano.h"
#include "gadget/constants.h"

namespace
{
struct peano_hilbert_data
{
  peanokey key;
  int index;
};

/*
struct peano_comparator
{
bool operator() (const peano_hilbert_data & a, const peano_hilbert_data & b)
{
  return a.key < b.key;
}
};
*/

const unsigned char rottable3[48][8] = {
    {36, 28, 25, 27, 10, 10, 25, 27}, {29, 11, 24, 24, 37, 11, 26, 26}, {8, 8, 25, 27, 30, 38, 25, 27},
    {9, 39, 24, 24, 9, 31, 26, 26},   {40, 24, 44, 32, 40, 6, 44, 6},   {25, 7, 33, 7, 41, 41, 45, 45},
    {4, 42, 4, 46, 26, 42, 34, 46},   {43, 43, 47, 47, 5, 27, 5, 35},   {33, 35, 36, 28, 33, 35, 2, 2},
    {32, 32, 29, 3, 34, 34, 37, 3},   {33, 35, 0, 0, 33, 35, 30, 38},   {32, 32, 1, 39, 34, 34, 1, 31},
    {24, 42, 32, 46, 14, 42, 14, 46}, {43, 43, 47, 47, 25, 15, 33, 15}, {40, 12, 44, 12, 40, 26, 44, 34},
    {13, 27, 13, 35, 41, 41, 45, 45}, {28, 41, 28, 22, 38, 43, 38, 22}, {42, 40, 23, 23, 29, 39, 29, 39},
    {41, 36, 20, 36, 43, 30, 20, 30}, {37, 31, 37, 31, 42, 40, 21, 21}, {28, 18, 28, 45, 38, 18, 38, 47},
    {19, 19, 46, 44, 29, 39, 29, 39}, {16, 36, 45, 36, 16, 30, 47, 30}, {37, 31, 37, 31, 17, 17, 46, 44},
    {12, 4, 1, 3, 34, 34, 1, 3},      {5, 35, 0, 0, 13, 35, 2, 2},      {32, 32, 1, 3, 6, 14, 1, 3},
    {33, 15, 0, 0, 33, 7, 2, 2},      {16, 0, 20, 8, 16, 30, 20, 30},   {1, 31, 9, 31, 17, 17, 21, 21},
    {28, 18, 28, 22, 2, 18, 10, 22},  {19, 19, 23, 23, 29, 3, 29, 11},  {9, 11, 12, 4, 9, 11, 26, 26},
    {8, 8, 5, 27, 10, 10, 13, 27},    {9, 11, 24, 24, 9, 11, 6, 14},    {8, 8, 25, 15, 10, 10, 25, 7},
    {0, 18, 8, 22, 38, 18, 38, 22},   {19, 19, 23, 23, 1, 39, 9, 39},   {16, 36, 20, 36, 16, 2, 20, 10},
    {37, 3, 37, 11, 17, 17, 21, 21},  {4, 17, 4, 46, 14, 19, 14, 46},   {18, 16, 47, 47, 5, 15, 5, 15},
    {17, 12, 44, 12, 19, 6, 44, 6},   {13, 7, 13, 7, 18, 16, 45, 45},   {4, 42, 4, 21, 14, 42, 14, 23},
    {43, 43, 22, 20, 5, 15, 5, 15},   {40, 12, 21, 12, 40, 6, 23, 6},   {13, 7, 13, 7, 41, 41, 22, 20}};

const unsigned char subpix3[48][8] = {
    {0, 7, 1, 6, 3, 4, 2, 5}, {7, 4, 6, 5, 0, 3, 1, 2}, {4, 3, 5, 2, 7, 0, 6, 1}, {3, 0, 2, 1, 4, 7, 5, 6}, {1, 0, 6, 7, 2, 3, 5, 4},
    {0, 3, 7, 4, 1, 2, 6, 5}, {3, 2, 4, 5, 0, 1, 7, 6}, {2, 1, 5, 6, 3, 0, 4, 7}, {6, 1, 7, 0, 5, 2, 4, 3}, {1, 2, 0, 3, 6, 5, 7, 4},
    {2, 5, 3, 4, 1, 6, 0, 7}, {5, 6, 4, 7, 2, 1, 3, 0}, {7, 6, 0, 1, 4, 5, 3, 2}, {6, 5, 1, 2, 7, 4, 0, 3}, {5, 4, 2, 3, 6, 7, 1, 0},
    {4, 7, 3, 0, 5, 6, 2, 1}, {6, 7, 5, 4, 1, 0, 2, 3}, {7, 0, 4, 3, 6, 1, 5, 2}, {0, 1, 3, 2, 7, 6, 4, 5}, {1, 6, 2, 5, 0, 7, 3, 4},
    {2, 3, 1, 0, 5, 4, 6, 7}, {3, 4, 0, 7, 2, 5, 1, 6}, {4, 5, 7, 6, 3, 2, 0, 1}, {5, 2, 6, 1, 4, 3, 7, 0}, {7, 0, 6, 1, 4, 3, 5, 2},
    {0, 3, 1, 2, 7, 4, 6, 5}, {3, 4, 2, 5, 0, 7, 1, 6}, {4, 7, 5, 6, 3, 0, 2, 1}, {6, 7, 1, 0, 5, 4, 2, 3}, {7, 4, 0, 3, 6, 5, 1, 2},
    {4, 5, 3, 2, 7, 6, 0, 1}, {5, 6, 2, 1, 4, 7, 3, 0}, {1, 6, 0, 7, 2, 5, 3, 4}, {6, 5, 7, 4, 1, 2, 0, 3}, {5, 2, 4, 3, 6, 1, 7, 0},
    {2, 1, 3, 0, 5, 6, 4, 7}, {0, 1, 7, 6, 3, 2, 4, 5}, {1, 2, 6, 5, 0, 3, 7, 4}, {2, 3, 5, 4, 1, 0, 6, 7}, {3, 0, 4, 7, 2, 1, 5, 6},
    {1, 0, 2, 3, 6, 7, 5, 4}, {0, 7, 3, 4, 1, 6, 2, 5}, {7, 6, 4, 5, 0, 1, 3, 2}, {6, 1, 5, 2, 7, 0, 4, 3}, {5, 4, 6, 7, 2, 3, 1, 0},
    {4, 3, 7, 0, 5, 2, 6, 1}, {3, 2, 0, 1, 4, 5, 7, 6}, {2, 5, 1, 6, 3, 4, 0, 7}};

}  // unnamed namespace

peanokey get_peanokey_offset(unsigned int j, int bits) /* this returns the peanokey for which  j << bits */
{
  peanokey key = {j, j, j};

  if(bits < BITS_FOR_POSITIONS)
    key.ls <<= bits;
  else
    key.ls = 0;

  int is_bits = bits - BITS_FOR_POSITIONS;

  if(is_bits <= -BITS_FOR_POSITIONS)
    key.is = 0;
  else if(is_bits <= 0)
    key.is >>= -is_bits;
  else if(is_bits < BITS_FOR_POSITIONS)
    key.is <<= is_bits;
  else
    key.is = 0;

  int hs_bits = bits - 2 * BITS_FOR_POSITIONS;

  if(hs_bits <= -BITS_FOR_POSITIONS)
    key.hs = 0;
  else if(hs_bits <= 0)
    key.hs >>= -hs_bits;
  else if(hs_bits < BITS_FOR_POSITIONS)
    key.hs <<= hs_bits;
  else
    key.hs = 0;

  return key;
}

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
 *  with x,y,z in the range between 0 and 2^bits-1.
 */
peanokey peano_hilbert_key(MyIntPosType x, MyIntPosType y, MyIntPosType z, int bits)
{
  unsigned char rotation = 0;
  peanokey key           = {0, 0, 0};

  for(MyIntPosType mask = ((MyIntPosType)1) << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key.hs <<= 3;
      key.hs |= (key.is & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      key.is <<= 3;
      key.is |= (key.ls & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      key.ls <<= 3;
      key.ls |= subpix3[rotation][pix];

      rotation = rottable3[rotation][pix];
    }

  return key;
}

unsigned char peano_incremental_key(unsigned char pix, unsigned char *rotation)
{
  unsigned char outpix = subpix3[*rotation][pix];
  *rotation            = rottable3[*rotation][pix];

  return outpix;
}

const unsigned char irottable3[48][8] = {
    {28, 27, 27, 10, 10, 25, 25, 36}, {29, 24, 24, 11, 11, 26, 26, 37}, {30, 25, 25, 8, 8, 27, 27, 38},
    {31, 26, 26, 9, 9, 24, 24, 39},   {32, 44, 44, 6, 6, 40, 40, 24},   {33, 45, 45, 7, 7, 41, 41, 25},
    {34, 46, 46, 4, 4, 42, 42, 26},   {35, 47, 47, 5, 5, 43, 43, 27},   {36, 33, 33, 2, 2, 35, 35, 28},
    {37, 34, 34, 3, 3, 32, 32, 29},   {38, 35, 35, 0, 0, 33, 33, 30},   {39, 32, 32, 1, 1, 34, 34, 31},
    {24, 42, 42, 14, 14, 46, 46, 32}, {25, 43, 43, 15, 15, 47, 47, 33}, {26, 40, 40, 12, 12, 44, 44, 34},
    {27, 41, 41, 13, 13, 45, 45, 35}, {41, 28, 28, 22, 22, 38, 38, 43}, {42, 29, 29, 23, 23, 39, 39, 40},
    {43, 30, 30, 20, 20, 36, 36, 41}, {40, 31, 31, 21, 21, 37, 37, 42}, {47, 38, 38, 18, 18, 28, 28, 45},
    {44, 39, 39, 19, 19, 29, 29, 46}, {45, 36, 36, 16, 16, 30, 30, 47}, {46, 37, 37, 17, 17, 31, 31, 44},
    {12, 1, 1, 34, 34, 3, 3, 4},      {13, 2, 2, 35, 35, 0, 0, 5},      {14, 3, 3, 32, 32, 1, 1, 6},
    {15, 0, 0, 33, 33, 2, 2, 7},      {0, 16, 16, 30, 30, 20, 20, 8},   {1, 17, 17, 31, 31, 21, 21, 9},
    {2, 18, 18, 28, 28, 22, 22, 10},  {3, 19, 19, 29, 29, 23, 23, 11},  {4, 11, 11, 26, 26, 9, 9, 12},
    {5, 8, 8, 27, 27, 10, 10, 13},    {6, 9, 9, 24, 24, 11, 11, 14},    {7, 10, 10, 25, 25, 8, 8, 15},
    {8, 22, 22, 38, 38, 18, 18, 0},   {9, 23, 23, 39, 39, 19, 19, 1},   {10, 20, 20, 36, 36, 16, 16, 2},
    {11, 21, 21, 37, 37, 17, 17, 3},  {19, 14, 14, 46, 46, 4, 4, 17},   {16, 15, 15, 47, 47, 5, 5, 18},
    {17, 12, 12, 44, 44, 6, 6, 19},   {18, 13, 13, 45, 45, 7, 7, 16},   {21, 4, 4, 42, 42, 14, 14, 23},
    {22, 5, 5, 43, 43, 15, 15, 20},   {23, 6, 6, 40, 40, 12, 12, 21},   {20, 7, 7, 41, 41, 13, 13, 22}};

const unsigned char ipixtable3[48][8] = {
    {1, 3, 7, 5, 4, 6, 2, 0}, {0, 2, 3, 1, 5, 7, 6, 4}, {4, 6, 2, 0, 1, 3, 7, 5}, {5, 7, 6, 4, 0, 2, 3, 1}, {3, 2, 6, 7, 5, 4, 0, 1},
    {2, 6, 7, 3, 1, 5, 4, 0}, {6, 7, 3, 2, 0, 1, 5, 4}, {7, 3, 2, 6, 4, 0, 1, 5}, {2, 0, 4, 6, 7, 5, 1, 3}, {6, 4, 5, 7, 3, 1, 0, 2},
    {7, 5, 1, 3, 2, 0, 4, 6}, {3, 1, 0, 2, 6, 4, 5, 7}, {0, 1, 5, 4, 6, 7, 3, 2}, {4, 0, 1, 5, 7, 3, 2, 6}, {5, 4, 0, 1, 3, 2, 6, 7},
    {1, 5, 4, 0, 2, 6, 7, 3}, {1, 0, 2, 3, 7, 6, 4, 5}, {0, 4, 6, 2, 3, 7, 5, 1}, {4, 5, 7, 6, 2, 3, 1, 0}, {5, 1, 3, 7, 6, 2, 0, 4},
    {7, 6, 4, 5, 1, 0, 2, 3}, {3, 7, 5, 1, 0, 4, 6, 2}, {2, 3, 1, 0, 4, 5, 7, 6}, {6, 2, 0, 4, 5, 1, 3, 7}, {0, 2, 6, 4, 5, 7, 3, 1},
    {4, 6, 7, 5, 1, 3, 2, 0}, {5, 7, 3, 1, 0, 2, 6, 4}, {1, 3, 2, 0, 4, 6, 7, 5}, {1, 0, 4, 5, 7, 6, 2, 3}, {0, 4, 5, 1, 3, 7, 6, 2},
    {4, 5, 1, 0, 2, 3, 7, 6}, {5, 1, 0, 4, 6, 2, 3, 7}, {3, 1, 5, 7, 6, 4, 0, 2}, {2, 0, 1, 3, 7, 5, 4, 6}, {6, 4, 0, 2, 3, 1, 5, 7},
    {7, 5, 4, 6, 2, 0, 1, 3}, {2, 3, 7, 6, 4, 5, 1, 0}, {6, 2, 3, 7, 5, 1, 0, 4}, {7, 6, 2, 3, 1, 0, 4, 5}, {3, 7, 6, 2, 0, 4, 5, 1},
    {5, 4, 6, 7, 3, 2, 0, 1}, {1, 5, 7, 3, 2, 6, 4, 0}, {0, 1, 3, 2, 6, 7, 5, 4}, {4, 0, 2, 6, 7, 3, 1, 5}, {3, 2, 0, 1, 5, 4, 6, 7},
    {2, 6, 4, 0, 1, 5, 7, 3}, {6, 7, 5, 4, 0, 1, 3, 2}, {7, 3, 1, 5, 4, 0, 2, 6},
};

void peano_hilbert_key_inverse(peanokey key, int bits, MyIntPosType *x, MyIntPosType *y, MyIntPosType *z)
{
  for(int i = bits; i < BITS_FOR_POSITIONS; i++)
    {
      key.hs <<= 3;
      key.hs |= (key.is & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      key.is <<= 3;
      key.is |= (key.ls & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      key.ls <<= 3;
    }

  int rot = 24;

  *x = *y = *z = 0;

  for(int i = 0; i < bits; i++)
    {
      unsigned int keypart = (key.hs & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      int quad = ipixtable3[rot][keypart];

      *x  = (*x << 1) + (quad >> 2);
      *y  = (*y << 1) + ((quad & 2) >> 1);
      *z  = (*z << 1) + (quad & 1);
      rot = irottable3[rot][keypart];

      key.hs <<= 3;
      key.hs |= (key.is & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      key.is <<= 3;
      key.is |= (key.ls & (~((~((MyIntPosType)0)) >> 3))) >> (BITS_FOR_POSITIONS - 3);

      key.ls <<= 3;
    }
}
peanokey operator+(const peanokey &a, const peanokey &b)
{
  peanokey c;

  c.ls = a.ls + b.ls;
  c.is = a.is + b.is;
  c.hs = a.hs + b.hs;

  if(c.is < a.is || c.is < b.is) /* overflow has occurred */
    {
      c.hs += 1;
    }

  if(c.ls < a.ls || c.ls < b.ls) /* overflow has occurred */
    {
      c.is += 1;
      if(c.is == 0) /* overflown again */
        c.hs += 1;
    }

  /* note: for hs we don't check for overflow explicitly as this would not be represented in the type anyhow */

  return c;
}
bool operator<(const peanokey &a, const peanokey &b)
{
  if(a.hs < b.hs)
    return true;
  else if(a.hs > b.hs)
    return false;
  else if(a.is < b.is)
    return true;
  else if(a.is > b.is)
    return false;
  else if(a.ls < b.ls)
    return true;
  else
    return false;
}
bool operator>=(const peanokey &a, const peanokey &b)
{
  if(a.hs < b.hs)
    return false;
  else if(a.hs > b.hs)
    return true;
  else if(a.is < b.is)
    return false;
  else if(a.is > b.is)
    return true;
  else if(a.ls < b.ls)
    return false;
  else
    return true;
}
