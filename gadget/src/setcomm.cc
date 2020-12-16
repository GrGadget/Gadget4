/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  setcomm.h
 *
 *  \brief implements a class providing basic information about the local MPI communicator
 */

#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

#include "gadget/setcomm.h"

void setcomm::initcomm(MPI_Comm Comm)
{
  Communicator = Comm;
  MPI_Comm_rank(Communicator, &ThisTask);
  MPI_Comm_size(Communicator, &NTask);

  for(PTask = 0; NTask > (1 << PTask); PTask++)
    ;
}

void setcomm::mpi_printf(const char *fmt, ...)
{
  if(ThisTask == 0)
    {
      va_list l;
      va_start(l, fmt);
      vprintf(fmt, l);
      va_end(l);
    }
}

bool setcomm::system_compare_hostname(const node_data &a, const node_data &b)
{
  int res = strcmp(a.name, b.name);
  if(res < 0)
    return true;
  if(res > 0)
    return false;
  return a.task < b.task;
}

bool setcomm::system_compare_first_task(const node_data &a, const node_data &b)
{
  if(a.first_task_in_this_node < b.first_task_in_this_node)
    return true;
  if(a.first_task_in_this_node > b.first_task_in_this_node)
    return false;
  return a.task < b.task;
}

void setcomm::determine_compute_nodes(void)
{
  int len, nodes, i, no, rank, first_index;

  MPI_Get_processor_name(loc_node.name, &len);
  loc_node.task = ThisTask;

  list_of_nodes = (node_data *)malloc(
      sizeof(node_data) * NTask); /* Note: Internal memory allocation routines are not yet available when this function is called */

  MPI_Allgather(&loc_node, sizeof(node_data), MPI_BYTE, list_of_nodes, sizeof(node_data), MPI_BYTE, Communicator);

  std::sort(list_of_nodes, list_of_nodes + NTask, system_compare_hostname);

  list_of_nodes[0].first_task_in_this_node = list_of_nodes[0].task;

  for(i = 1, nodes = 1; i < NTask; i++)
    {
      if(strcmp(list_of_nodes[i].name, list_of_nodes[i - 1].name) != 0)
        {
          list_of_nodes[i].first_task_in_this_node = list_of_nodes[i].task;
          nodes++;
        }
      else
        list_of_nodes[i].first_task_in_this_node = list_of_nodes[i - 1].first_task_in_this_node;
    }

  std::sort(list_of_nodes, list_of_nodes + NTask, system_compare_first_task);

  for(i = 0; i < NTask; i++)
    list_of_nodes[i].tasks_in_node = 0;

  for(i = 0, no = 0, rank = 0, first_index = 0; i < NTask; i++)
    {
      if(i ? list_of_nodes[i].first_task_in_this_node != list_of_nodes[i - 1].first_task_in_this_node : 0)
        {
          no++;
          rank        = 0;
          first_index = i;
        }

      list_of_nodes[i].first_index  = first_index;
      list_of_nodes[i].this_node    = no;
      list_of_nodes[i].rank_in_node = rank++;
      list_of_nodes[first_index].tasks_in_node++;
    }

  int max_count = 0;
  int min_count = (1 << 30);

  for(i = 0; i < NTask; i++)
    {
      list_of_nodes[i].tasks_in_node = list_of_nodes[list_of_nodes[i].first_index].tasks_in_node;

      if(list_of_nodes[i].tasks_in_node > max_count)
        max_count = list_of_nodes[i].tasks_in_node;
      if(list_of_nodes[i].tasks_in_node < min_count)
        min_count = list_of_nodes[i].tasks_in_node;
    }

  std::sort(list_of_nodes, list_of_nodes + NTask, system_compare_task);

  TasksInThisNode = list_of_nodes[ThisTask].tasks_in_node;
  RankInThisNode  = list_of_nodes[ThisTask].rank_in_node;

  ThisNode = list_of_nodes[ThisTask].this_node;

  NumNodes        = nodes;
  MinTasksPerNode = min_count;
  MaxTasksPerNode = max_count;

  free(list_of_nodes);
}
