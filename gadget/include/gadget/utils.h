#pragma once

void subdivide_evenly(long long N, int pieces, int index_bin, long long *first, int *count);
void subdivide_evenly(int N, int pieces, int index, int *first, int *count);
void subdivide_evenly_get_bin(int N, int pieces, int index, int *bin);
int my_fls(unsigned int x);

template <typename T>
inline T square(T const value)
{
  return value * value;
}
