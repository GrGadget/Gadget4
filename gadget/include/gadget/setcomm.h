/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  setcomm.h
 *
 *  \brief implements a class providing basic information about the local MPI communicator
 */

#ifndef SETCOMM_H
#define SETCOMM_H
#include <mpi.h>

namespace gadget{
class setcomm
{
 public:
  setcomm(MPI_Comm Comm) { initcomm(Comm); }
  setcomm(const char *str)
  {
    /* do nothing in this case, because we need to delay the initialization until MPI_Init has been executed */
  }

  MPI_Comm Communicator;
  int NTask;
  int ThisTask;
  int PTask;

  int ThisNode;        /**< the rank of the current compute node  */
  int NumNodes = 0;    /**< the number of compute nodes used  */
  int TasksInThisNode; /**< number of MPI tasks on  current compute node */
  int RankInThisNode;  /**< rank of the MPI task on the current compute node */
  int MinTasksPerNode; /**< the minimum number of MPI tasks that is found on any of the nodes  */
  int MaxTasksPerNode; /**< the maximum number of MPI tasks that is found on any of the nodes  */
  long long MemoryOnNode;
  long long SharedMemoryOnNode;

  void initcomm(MPI_Comm Comm);

  void mpi_printf(const char *fmt, ...);

 private:
  struct node_data
  {
    int task, this_node, first_task_in_this_node;
    int first_index, rank_in_node, tasks_in_node;
    char name[MPI_MAX_PROCESSOR_NAME];
  };
  node_data loc_node, *list_of_nodes;

  static bool system_compare_hostname(const node_data &a, const node_data &b);

  static bool system_compare_first_task(const node_data &a, const node_data &b);

  static bool system_compare_task(const node_data &a, const node_data &b) { return a.task < b.task; }

 public:
  void determine_compute_nodes(void);
};
}
#endif
