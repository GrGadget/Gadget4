#include "gadget/utils.h"
#include <climits>          // INT_MAX
#include "gadget/macros.h"  // Terminate

namespace gadget{ 
void subdivide_evenly(int N, int pieces, int index_bin, int *first, int *count)
{
  int nbase      = N / pieces;
  int additional = N % pieces;
  *first         = index_bin * nbase + ((index_bin < additional) ? index_bin : additional);
  *count         = nbase + (index_bin < additional);
}

void subdivide_evenly_get_bin(int N, int pieces, int index, int *bin)
{
  int nbase      = N / pieces;
  int additional = N % pieces;

  if(index < additional * (nbase + 1))
    *bin = index / (nbase + 1);
  else
    *bin = (index - additional) / nbase;
}

void subdivide_evenly(long long N, int pieces, int index_bin, long long *first, int *count)
{
  if(N / pieces > INT_MAX)
    Terminate("overflow");

  int nbase      = N / pieces;
  int additional = N % pieces;
  *first         = index_bin * ((long long)nbase) + ((index_bin < additional) ? index_bin : additional);
  *count         = nbase + (index_bin < additional);
}

/* the following function finds the last significant bit, as in the linux kernel */
int my_fls(unsigned int x)
{
  int r = 32;

  if(!x)
    return 0;
  if(!(x & 0xffff0000u))
    {
      x <<= 16;
      r -= 16;
    }
  if(!(x & 0xff000000u))
    {
      x <<= 8;
      r -= 8;
    }
  if(!(x & 0xf0000000u))
    {
      x <<= 4;
      r -= 4;
    }
  if(!(x & 0xc0000000u))
    {
      x <<= 2;
      r -= 2;
    }
  if(!(x & 0x80000000u))
    {
      x <<= 1;
      r -= 1;
    }
  return r;
}
}
