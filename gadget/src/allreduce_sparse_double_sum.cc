/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  allreduce_sparse_double_sum.cc
 *
 *  \brief implementation of a reduction operation for sparsely populated data
 */

#include <vector>

extern template class std::vector<int>;
extern template class std::vector<double>;

#include "gadgetconfig.h"

#include "gadget/mpi_utils.h"

namespace gadget{ 
void allreduce_sparse_double_sum(double *loc, double *glob, int N, MPI_Comm Communicator)
{
  int ntask, thistask, ptask;
  MPI_Comm_size(Communicator, &ntask);
  MPI_Comm_rank(Communicator, &thistask);

  for(ptask = 0; ntask > (1 << ptask); ptask++)
    ;

  std::vector<int> send_count(ntask), recv_count(ntask), send_offset(ntask), recv_offset(ntask), blocksize(ntask);

  int blk     = N / ntask;
  int rmd     = N - blk * ntask; /* remainder */
  int pivot_n = rmd * (blk + 1);

  int loc_first_n = 0;
  for(int task = 0; task < ntask; task++)
    {
      if(task < rmd)
        blocksize[task] = blk + 1;
      else
        blocksize[task] = blk;

      if(task < thistask)
        loc_first_n += blocksize[task];
    }

  std::vector<double> loc_data(blocksize[thistask], 0.0);

  for(int j = 0; j < ntask; j++)
    send_count[j] = 0;

  /* find for each non-zero element the processor where it should go for being summed */
  for(int n = 0; n < N; n++)
    {
      if(loc[n] != 0)
        {
          int task;
          if(n < pivot_n)
            task = n / (blk + 1);
          else
            task = rmd + (n - pivot_n) / blk; /* note: if blk=0, then this case can not occur */

          send_count[task]++;
        }
    }

  MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, Communicator);

  int nimport = 0, nexport = 0;

  recv_offset[0] = 0, send_offset[0] = 0;

  for(int j = 0; j < ntask; j++)
    {
      nexport += send_count[j];
      nimport += recv_count[j];
      if(j > 0)
        {
          send_offset[j] = send_offset[j - 1] + send_count[j - 1];
          recv_offset[j] = recv_offset[j - 1] + recv_count[j - 1];
        }
    }

  struct ind_data
  {
    int n;
    double val;
  };
  std::vector<ind_data> export_data(nexport), import_data(nimport);

  for(int j = 0; j < ntask; j++)
    send_count[j] = 0;

  for(int n = 0; n < N; n++)
    {
      if(loc[n] != 0)
        {
          int task;

          if(n < pivot_n)
            task = n / (blk + 1);
          else
            task = rmd + (n - pivot_n) / blk; /* note: if blk=0, then this case can not occur */

          int index              = send_offset[task] + send_count[task]++;
          export_data[index].n   = n;
          export_data[index].val = loc[n];
        }
    }

  for(int ngrp = 0; ngrp < (1 << ptask); ngrp++) /* note: here we also have a transfer from each task to itself (for ngrp=0) */
    {
      int recvTask = thistask ^ ngrp;
      if(recvTask < ntask)
        if(send_count[recvTask] > 0 || recv_count[recvTask] > 0)
          MPI_Sendrecv(&export_data[send_offset[recvTask]], send_count[recvTask] * sizeof(ind_data), MPI_BYTE, recvTask, TAG_DENS_B,
                       &import_data[recv_offset[recvTask]], recv_count[recvTask] * sizeof(ind_data), MPI_BYTE, recvTask, TAG_DENS_B,
                       Communicator, MPI_STATUS_IGNORE);
    }

  for(int i = 0; i < nimport; i++)
    {
      int j = import_data[i].n - loc_first_n;

      if(j < 0 || j >= blocksize[thistask])
        Terminate("j=%d < 0 || j>= blocksize[thistask]=%d", j, blocksize[thistask]);

      loc_data[j] += import_data[i].val;
    }

  /* now share the cost data across all processors */
  std::vector<int> bytecounts(ntask), byteoffset(ntask);

  for(int task = 0; task < ntask; task++)
    bytecounts[task] = blocksize[task] * sizeof(double);

  byteoffset[0] = 0;
  for(int task = 1; task < ntask; task++)
    byteoffset[task] = byteoffset[task - 1] + bytecounts[task - 1];

  MPI_Allgatherv(loc_data.data(), bytecounts[thistask], MPI_BYTE, glob, bytecounts.data(), byteoffset.data(), MPI_BYTE, Communicator);
}
}
