/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 ******************************************************************************/

/*! \file  myalltoall.cc
 *
 *  \brief a simple wrapper around MPI_Alltoallv that can deal with data in individual sends that are very big
 */

#include <cstring>  // memcpy
#include <vector>

extern template class std::vector<int>;
#include "gadgetconfig.h"

#include "gadget/mpi_utils.h"

namespace gadget{

#define PCHAR(a) ((char *)a)

/* This method prepares an Alltoallv computation.
   sendcnt: must have as many entries as there are Tasks in comm
            must be set
   recvcnt: must have as many entries as there are Tasks in comm
            will be set on return
   rdispls: must have as many entries as there are Tasks in comm, or be NULL
            if not NULL, will be set on return
   method:  use standard Alltoall() approach or one-sided approach
   returns: number of entries needed in the recvbuf */
int myMPI_Alltoallv_new_prep(int *sendcnt, int *recvcnt, int *rdispls, MPI_Comm comm, int method)
{
  int rank, nranks;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);

  if(method == 0 || method == 1)
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, comm);
  else if(method == 10)
    {
      for(int i = 0; i < nranks; ++i)
        recvcnt[i] = 0;
      recvcnt[rank] = sendcnt[rank];  // local communication
      MPI_Win win;
      MPI_Win_create(recvcnt, nranks * sizeof(MPI_INT), sizeof(MPI_INT), MPI_INFO_NULL, comm, &win);
      MPI_Win_fence(0, win);
      for(int i = 1; i < nranks; ++i)  // remote communication
        {
          int tgt = (rank + i) % nranks;
          if(sendcnt[tgt] != 0)
            MPI_Put(&sendcnt[tgt], 1, MPI_INT, tgt, rank, 1, MPI_INT, win);
        }
      MPI_Win_fence(0, win);
      MPI_Win_free(&win);
    }
  else
    Terminate("bad communication method");

  int total = 0;
  for(int i = 0; i < nranks; ++i)
    {
      if(rdispls)
        rdispls[i] = total;
      total += recvcnt[i];
    }
  return total;
}

void myMPI_Alltoallv_new(void *sendbuf, int *sendcnt, int *sdispls, MPI_Datatype sendtype, void *recvbuf, int *recvcnt, int *rdispls,
                         MPI_Datatype recvtype, MPI_Comm comm, int method)
{
  int rank, nranks, itsz;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);
  MPI_Type_size(sendtype, &itsz);
  size_t tsz = itsz;  // to enforce size_t data type in later computations

  if(method == 0)  // standard Alltoallv
    MPI_Alltoallv(sendbuf, sendcnt, sdispls, sendtype, recvbuf, recvcnt, rdispls, recvtype, comm);
  else if(method == 1)  // blocking sendrecv
    {
      if(sendtype != recvtype)
        Terminate("bad MPI communication types");
      int lptask = 1;
      while(lptask < nranks)
        lptask <<= 1;
      int tag = 42;
      MPI_Status status;

      if(recvcnt[rank] > 0)  // local communication
        memcpy(PCHAR(recvbuf) + tsz * rdispls[rank], PCHAR(sendbuf) + tsz * sdispls[rank], tsz * recvcnt[rank]);

      for(int ngrp = 1; ngrp < lptask; ngrp++)
        {
          int otask = rank ^ ngrp;
          if(otask < nranks)
            if(sendcnt[otask] > 0 || recvcnt[otask] > 0)
              MPI_Sendrecv(PCHAR(sendbuf) + tsz * sdispls[otask], sendcnt[otask], sendtype, otask, tag,
                           PCHAR(recvbuf) + tsz * rdispls[otask], recvcnt[otask], recvtype, otask, tag, comm, &status);
        }
    }
  else if(method == 2)  // asynchronous communication
    {
      if(sendtype != recvtype)
        Terminate("bad MPI communication types");
      int lptask = 1;
      while(lptask < nranks)
        lptask <<= 1;
      int tag = 42;

      std::vector<MPI_Request> requests(2 * nranks);
      int n_requests = 0;

      if(recvcnt[rank] > 0)  // local communication
        memcpy(PCHAR(recvbuf) + tsz * rdispls[rank], PCHAR(sendbuf) + tsz * sdispls[rank], tsz * recvcnt[rank]);

      for(int ngrp = 1; ngrp < lptask; ngrp++)
        {
          int otask = rank ^ ngrp;
          if(otask < nranks)
            if(recvcnt[otask] > 0)
              MPI_Irecv(PCHAR(recvbuf) + tsz * rdispls[otask], recvcnt[otask], recvtype, otask, tag, comm, &requests[n_requests++]);
        }

      for(int ngrp = 1; ngrp < lptask; ngrp++)
        {
          int otask = rank ^ ngrp;
          if(otask < nranks)
            if(sendcnt[otask] > 0)
              MPI_Issend(PCHAR(sendbuf) + tsz * sdispls[otask], sendcnt[otask], sendtype, otask, tag, comm, &requests[n_requests++]);
        }

      MPI_Waitall(n_requests, requests.data(), MPI_STATUSES_IGNORE);
    }
  else if(method == 10)
    {
      if(sendtype != recvtype)
        Terminate("bad MPI communication types");
      std::vector<int> disp_at_sender(nranks);
      disp_at_sender[rank] = sdispls[rank];
      MPI_Win win;
      MPI_Win_create(sdispls, nranks * sizeof(MPI_INT), sizeof(MPI_INT), MPI_INFO_NULL, comm, &win);
      MPI_Win_fence(0, win);
      for(int i = 1; i < nranks; ++i)
        {
          int tgt = (rank + i) % nranks;
          if(recvcnt[tgt] != 0)
            MPI_Get(&disp_at_sender[tgt], 1, MPI_INT, tgt, rank, 1, MPI_INT, win);
        }
      MPI_Win_fence(0, win);
      MPI_Win_free(&win);
      if(recvcnt[rank] > 0)  // first take care of local communication
        memcpy(PCHAR(recvbuf) + tsz * rdispls[rank], PCHAR(sendbuf) + tsz * sdispls[rank], tsz * recvcnt[rank]);
      MPI_Win_create(sendbuf, (sdispls[nranks - 1] + sendcnt[nranks - 1]) * tsz, tsz, MPI_INFO_NULL, comm, &win);
      MPI_Win_fence(0, win);
      for(int i = 1; i < nranks; ++i)  // now the rest, start with right neighbour
        {
          int tgt = (rank + i) % nranks;
          if(recvcnt[tgt] != 0)
            MPI_Get(PCHAR(recvbuf) + tsz * rdispls[tgt], recvcnt[tgt], sendtype, tgt, disp_at_sender[tgt], recvcnt[tgt], sendtype,
                    win);
        }
      MPI_Win_fence(0, win);
      MPI_Win_free(&win);
    }
  else
    Terminate("bad communication method");
}

void myMPI_Alltoallv(void *sendb, size_t *sendcounts, size_t *sdispls, void *recvb, size_t *recvcounts, size_t *rdispls, int len,
                     int big_flag, MPI_Comm comm)
{
  char *sendbuf = (char *)sendb;
  char *recvbuf = (char *)recvb;

  if(big_flag == 0)
    {
      int ntask;
      MPI_Comm_size(comm, &ntask);

      std::vector<int> scount(ntask), rcount(ntask), soff(ntask), roff(ntask);

      for(int i = 0; i < ntask; i++)
        {
          scount[i] = sendcounts[i] * len;
          rcount[i] = recvcounts[i] * len;
          soff[i]   = sdispls[i] * len;
          roff[i]   = rdispls[i] * len;
        }

      MPI_Alltoallv(sendbuf, scount.data(), soff.data(), MPI_BYTE, recvbuf, rcount.data(), roff.data(), MPI_BYTE, comm);
    }
  else
    {
      /* here we definitely have some large messages. We default to the
       * pair-wise protocol, which should be most robust anyway.
       */
      int ntask, thistask, ptask;
      MPI_Comm_size(comm, &ntask);
      MPI_Comm_rank(comm, &thistask);

      for(ptask = 0; ntask > (1 << ptask); ptask++)
        ;

      for(int ngrp = 0; ngrp < (1 << ptask); ngrp++)
        {
          int target = thistask ^ ngrp;

          if(target < ntask)
            {
              if(sendcounts[target] > 0 || recvcounts[target] > 0)
                myMPI_Sendrecv(sendbuf + sdispls[target] * len, sendcounts[target] * len, MPI_BYTE, target, TAG_PDATA + ngrp,
                               recvbuf + rdispls[target] * len, recvcounts[target] * len, MPI_BYTE, target, TAG_PDATA + ngrp, comm,
                               MPI_STATUS_IGNORE);
            }
        }
    }
}

void my_int_MPI_Alltoallv(void *sendb, int *sendcounts, int *sdispls, void *recvb, int *recvcounts, int *rdispls, int len,
                          int big_flag, MPI_Comm comm)
{
  char *sendbuf = (char *)sendb;
  char *recvbuf = (char *)recvb;

  if(big_flag == 0)
    {
      int ntask;
      MPI_Comm_size(comm, &ntask);

      std::vector<int> scount(ntask), rcount(ntask), soff(ntask), roff(ntask);

      for(int i = 0; i < ntask; i++)
        {
          scount[i] = sendcounts[i] * len;
          rcount[i] = recvcounts[i] * len;
          soff[i]   = sdispls[i] * len;
          roff[i]   = rdispls[i] * len;
        }

      MPI_Alltoallv(sendbuf, scount.data(), soff.data(), MPI_BYTE, recvbuf, rcount.data(), roff.data(), MPI_BYTE, comm);
    }
  else
    {
      /* here we definitely have some large messages. We default to the
       * pair-wise protocoll, which should be most robust anyway.
       */
      int ntask, thistask, ptask;
      MPI_Comm_size(comm, &ntask);
      MPI_Comm_rank(comm, &thistask);

      for(ptask = 0; ntask > (1 << ptask); ptask++)
        ;

      for(int ngrp = 0; ngrp < (1 << ptask); ngrp++)
        {
          int target = thistask ^ ngrp;

          if(target < ntask)
            {
              if(sendcounts[target] > 0 || recvcounts[target] > 0)
                myMPI_Sendrecv(sendbuf + sdispls[target] * len, sendcounts[target] * len, MPI_BYTE, target, TAG_PDATA + ngrp,
                               recvbuf + rdispls[target] * len, recvcounts[target] * len, MPI_BYTE, target, TAG_PDATA + ngrp, comm,
                               MPI_STATUS_IGNORE);
            }
        }
    }
}

}
