/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  sums_and_minmax.cc
 *
 *  \brief some simple extensions of MPI-collectives
 */

#include <vector>
extern template class std::vector<long long>;

#include "gadgetconfig.h"

#include "gadget/mpi_utils.h"

namespace gadget{ 
void minimum_large_ints(int n, long long *src, long long *res, MPI_Comm comm)
{
  if(src == res)
    MPI_Allreduce(MPI_IN_PLACE, res, n, MPI_LONG_LONG, MPI_MIN, comm);
  else
    MPI_Allreduce(src, res, n, MPI_LONG_LONG, MPI_MIN, comm);
}

/*
    TODO:
    remove the need to use a temporary memory buffer numlist
*/
void sumup_large_ints(int n, int *src, long long *res, MPI_Comm comm)
{
  std::vector<long long> numlist(src, src + n);
  MPI_Allreduce(numlist.data(), res, n, MPI_LONG_LONG, MPI_SUM, comm);
}

void sumup_longs(int n, long long *src, long long *res, MPI_Comm comm)
{
  if(src == res)
    MPI_Allreduce(MPI_IN_PLACE, res, n, MPI_LONG_LONG, MPI_SUM, comm);
  else
    MPI_Allreduce(src, res, n, MPI_LONG_LONG, MPI_SUM, comm);
}
}
