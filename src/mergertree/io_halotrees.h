/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  io_halotrees.h
 *
 *  \brief definitions of a class for storing the halo trees
 */

#ifndef HALOTREES_IO_H
#define HALOTREES_IO_H

#include "gadgetconfig.h"

#ifdef MERGERTREE

#include "../data/allvars.h"
#include "../data/mymalloc.h"
#include "../fof/fof.h"
#include "../io/io.h"
#include "../logs/logs.h"
#include "../main/simulation.h"
#include "../mergertree/mergertree.h"
#include "../sort/parallel_sort.h"
#include "../subfind/subfind.h"
#include "../system/system.h"
#include "gadget/dtypes.h"
#include "gadget/hdf5_util.h"
#include "gadget/mpi_utils.h"

namespace gaget{
class halotrees_io : public IO_Def
{
 private:
  mergertree *MergerTree;

 public:
  halotrees_io(mergertree *MergerTree_ptr, MPI_Comm comm, int format);

  void halotrees_save_trees(void);

  /* supplied virtual functions */
  void fill_file_header(int writeTask, int lastTask, long long *nloc_part, long long *npart);
  void read_file_header(const char *fname, int filenr, int readTask, int lastTask, long long *nloc_part, long long *npart,
                        int *nstart);
  void get_datagroup_name(int grnr, char *gname);
  void write_header_fields(hid_t);
  void read_header_fields(const char *fname);
  void read_increase_numbers(int type, int n_for_this_task);
  int get_filenr_from_header(void);
  void set_filenr_in_header(int);
  void *get_base_address_of_structure(enum arrays array, int index);
  int get_type_of_element(int index);
  void set_type_of_element(int index, int type);

  /** Header for the standard file format.
   */

  struct io_header
  {
    long long Nhalos;
    long long TotNhalos;

    long long Ntrees;
    long long TotNtrees;

    int num_files;
    int lastsnapshotnr;
  };
  io_header header;
};
}
#endif

#endif /* HALOTREES_IO_H */
