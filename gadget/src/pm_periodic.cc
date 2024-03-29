/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  pm_periodic.cc
 *
 *  \brief routines for periodic PM-force calculation
 */

#include "gadgetconfig.h"

#include <fftw3.h>
#include <mpi.h>
#include <sys/stat.h>  // mkdir
#include <algorithm>   // sort, fill
#include <cmath>       // sin, exp
#include <cstring>     // memcpy
#include <numeric>     // accumulate
#include <tuple>
#include <vector>

#include "gadget/constants.h"         // NTYPES
#include "gadget/dtypes.h"            // MyFloat
#include "gadget/macros.h"            // Terminate
#include "gadget/mpi_utils.h"         // myMPI_Sendrecv
#include "gadget/particle_handler.h"  // particle_handler
#include "gadget/pm_periodic.h"

extern template class std::vector<gadget::MyFloat>;
extern template class std::vector<size_t>;

namespace gadget{
/*!
 * These routines support two different strategies for doing the particle data exchange to assemble the density field
 * and to read out the forces and potentials:
 *
 * The default scheme sends the particle positions to the target slabs, and bins them there. This works usually well for
 * homogenuously loaded boxes, but can be problematic for zoom-in runs. In the latter case,  PM_ZOOM_OPTIMIZED can be
 * activated, where the data is binned on the originating processor followed by assembly of the binned density field.
 *
 * In addition, the routines can be either used with a slab-based FFT (as is traditionally done in FFTW), or with a
 * column-based FFT. The latter requires more communication and is hence usually slower than the slab-based one.
 * But if the number of MPI ranks exceeds the number of cells per dimension, then the column-based one can still scale
 * and offers a balanced memory consumption, whereas this is not the case for the slab-based approach. To select the
 * column-based FFT, the switch FFT_COLUMN_BASED can be activated.
 *
 * The switches PM_ZOOM_OPTIMIZED and FFT_COLUMN_BASED may also be combined, such that there are 4 main modes of how the
 * PM routines may operate.
 *
 * It is also possible to use non-cubical boxes, by means of setting one or several of the LONG_X, LONG_Y, and LONG_Z
 * options in the config file. The values need to be integers, and then BoxSize is stretched by that factor in the
 * corresponding dimension.
 *
 * Finally, one may also use the TreePM routine for simulations where gravity is perdiodic only in two spatial dimensions.
 * The non-periodic dimension is selected via the GRAVITY_TALLBOX flag. Also in this case, arbitrarily stretched boxes can
 * be used, and one can use PM_ZOOM_OPTIMIZED and/or FFT_COLUMN_BASED if desired.
 *
 * If eight times the particle load per processor exceeds 2^31 ~ 2 billion, one should activate NUMPART_PER_TASK_LARGE.
 * The code will check this condition and terminate if this is violated, so there should hopefully be no severe risk
 * to accidentally forget this.
 */

#ifdef LEAN
MyFloat pm_periodic::partbuf::Mass;
#endif

/*! \brief This routine generates the FFTW-plans to carry out the FFTs later on.
 *
 *  Some auxiliary variables for bookkeeping are also initialized.
 */
void pm_periodic::pm_init_periodic(particle_handler *Sp_ptr, double boxsize, double asmth)
{
  asmth2  = asmth * asmth;
  BoxSize = boxsize;
  Sp.reset(Sp_ptr);

  /* Set up the FFTW-3 plan files. */
  int ndimx[1] = {Ngrid[0]}; /* dimension of the 1D transforms */
  int ndimy[1] = {Ngrid[1]}; /* dimension of the 1D transforms */
  int ndimz[1] = {Ngrid[2]}; /* dimension of the 1D transforms */

  int max_Ngrid2 = 2 * (std::max<int>(std::max<int>(Ngrid[0], Ngrid[1]), Ngrid[2]) / 2 + 1);

  /* temporarily allocate some arrays to make sure that out-of-place plans are created */
  rhogrid.resize(max_Ngrid2);
  std::fill(rhogrid.begin(), rhogrid.end(), 0);
  forcegrid.resize(max_Ngrid2);

#ifdef DOUBLEPRECISION_FFTW
  int alignflag = 0;
#else
  /* for single precision, the start of our FFT columns is presently only guaranteed to be 8-byte aligned */
  int alignflag = FFTW_UNALIGNED;
#endif

  forward_plan_zdir = FFTW(plan_many_dft_r2c)(1, ndimz, 1, rhogrid.data(), 0, 1, Ngrid2, (fft_complex *)forcegrid.data(), 0, 1, Ngridz,
                                              FFTW_ESTIMATE | FFTW_DESTROY_INPUT | alignflag);

#ifndef FFT_COLUMN_BASED
  int stride = Ngridz;
#else
  int stride    = 1;
#endif

  forward_plan_ydir =
      FFTW(plan_many_dft)(1, ndimy, 1, (fft_complex *)rhogrid.data(), 0, stride, Ngridz * Ngrid[1], (fft_complex *)forcegrid.data(), 0,
                          stride, Ngridz * Ngrid[1], FFTW_FORWARD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | alignflag);

  forward_plan_xdir =
      FFTW(plan_many_dft)(1, ndimx, 1, (fft_complex *)rhogrid.data(), 0, stride, Ngridz * Ngrid[0], (fft_complex *)forcegrid.data(), 0,
                          stride, Ngridz * Ngrid[0], FFTW_FORWARD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | alignflag);

  backward_plan_xdir =
      FFTW(plan_many_dft)(1, ndimx, 1, (fft_complex *)rhogrid.data(), 0, stride, Ngridz * Ngrid[0], (fft_complex *)forcegrid.data(), 0,
                          stride, Ngridz * Ngrid[0], FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | alignflag);

  backward_plan_ydir =
      FFTW(plan_many_dft)(1, ndimy, 1, (fft_complex *)rhogrid.data(), 0, stride, Ngridz * Ngrid[1], (fft_complex *)forcegrid.data(), 0,
                          stride, Ngridz * Ngrid[1], FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | alignflag);

  backward_plan_zdir = FFTW(plan_many_dft_c2r)(1, ndimz, 1, (fft_complex *)rhogrid.data(), 0, 1, Ngridz, forcegrid.data(), 0, 1,
                                               Ngrid2, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | alignflag);

#ifndef FFT_COLUMN_BASED

  maxfftsize = std::max<int>(largest_x_slab * Ngrid[1], largest_y_slab * Ngrid[0]) * ((size_t)Ngrid2);

#else

  maxfftsize = max_datasize;

#endif

#if defined(GRAVITY_TALLBOX)
  kernel.reset(new fft_real[maxfftsize]);
  fft_of_kernel = (std::complex<fft_real> *)kernel.get();

  pmforce_setup_tallbox_kernel();
#endif
}

/* Below, the two functions
 *
 *           pmforce_ ...... _prepare_density()
 * and
 *           pmforce_ ...... _readout_forces_or_potential(int dim)
 *
 * are defined in two different versions, one that works better for uniform
 * simulations, the other for zoom-in runs. Only one of the two sets is used,
 * depending on the setting of PM_ZOOM_OPTIMIZED.
 */

#ifdef PM_ZOOM_OPTIMIZED

void pm_periodic::pmforce_zoom_optimized_prepare_density(int mode, int *typelist, std::vector<part_slab_data> &part,
                                                         std::vector<large_array_offset> &localfield_globalindex,
                                                         std::vector<fft_real> &localfield_data)
{
  int level, recvTask;
  MPI_Status status;

  // particle_data *P = Sp->P;

  const int NSource                    = Sp->size();
  const large_numpart_type num_on_grid = ((large_numpart_type)NSource) * 8;
  part.resize(num_on_grid);
  std::vector<large_numpart_type> part_sortindex(num_on_grid);

#ifdef FFT_COLUMN_BASED
  int columns         = Ngrid[0] * Ngrid[1];
  int avg             = (columns - 1) / NTask + 1;
  int exc             = NTask * avg - columns;
  int tasklastsection = NTask - exc;
  int pivotcol        = tasklastsection * avg;
#endif

  /* determine the cells each particle accesses */
  for(int idx = 0; idx < NSource; idx++)
    {
      int fact                         = pw_fold_factor(mode);
      auto [slab_x, slab_y, slab_z]    = grid_coordinates(Sp->get_IntPosition(idx), fact);
      large_numpart_type index_on_grid = ((large_numpart_type)idx) * 8;

      for(int xx = 0; xx < 2; xx++)
        for(int yy = 0; yy < 2; yy++)
          for(int zz = 0; zz < 2; zz++)
            {
              int slab_xx = (slab_x + xx) % Ngrid[0];
              int slab_yy = (slab_y + yy) % Ngrid[1];
              int slab_zz = (slab_z + zz) % Ngrid[2];

              large_array_offset offset = FI(slab_xx, slab_yy, slab_zz);

              part[index_on_grid].partindex   = (idx << 3) + (xx << 2) + (yy << 1) + zz;
              part[index_on_grid].globalindex = offset;
              part_sortindex[index_on_grid]   = index_on_grid;
              index_on_grid++;
            }
    }

  /* note: num_on_grid will be  8 times larger than the particle number, but num_field_points will generally be much smaller */

  /* bring the part-field into the order of the accessed cells. This allows the removal of duplicates */

  std::sort(part_sortindex.begin(), part_sortindex.end(),
            [&part](const large_numpart_type a, const large_numpart_type b) { return part[a].globalindex < part[b].globalindex; });

  large_array_offset num_field_points = num_on_grid > 0 ? 1 : 0;
  /* determine the number of unique field points */
  for(large_numpart_type i = 1; i < num_on_grid; i++)
    {
      if(part[part_sortindex[i]].globalindex != part[part_sortindex[i - 1]].globalindex)
        num_field_points++;
    }

  /* allocate the local field */
  localfield_globalindex.resize(num_field_points);
  localfield_data.resize(num_field_points);

  for(int i = 0; i < NTask; i++)
    {
      Rcvpm_offset[i] = 0;
      Sndpm_count[i]  = 0;
    }

  /* establish the cross link between the part[ ]-array and the local list of
   * mesh points. Also, count on which CPU the needed field points are stored.
   */
  num_field_points = 0;
  for(large_numpart_type i = 0; i < num_on_grid; i++)
    {
      if(i > 0)
        if(part[part_sortindex[i]].globalindex != part[part_sortindex[i - 1]].globalindex)
          num_field_points++;

      part[part_sortindex[i]].localindex = num_field_points;

      if(i > 0)
        if(part[part_sortindex[i]].globalindex == part[part_sortindex[i - 1]].globalindex)
          continue;

      localfield_globalindex[num_field_points] = part[part_sortindex[i]].globalindex;

#ifndef FFT_COLUMN_BASED
      int slab = part[part_sortindex[i]].globalindex / (Ngrid[1] * Ngrid2);
      int task = slab_to_task[slab];
#else
      int task, column = part[part_sortindex[i]].globalindex / (Ngrid2);

      if(column < pivotcol)
        task = column / avg;
      else
        task = (column - pivotcol) / (avg - 1) + tasklastsection;
#endif

      if(Sndpm_count[task] == 0)
        Rcvpm_offset[task] = num_field_points;

      Sndpm_count[task]++;
    }
  num_field_points++;

  Sndpm_offset[0] = 0;
  for(int i = 1; i < NTask; i++)
    Sndpm_offset[i] = Sndpm_offset[i - 1] + Sndpm_count[i - 1];

  /* now bin the local particle data onto the mesh list */
  for(large_numpart_type i = 0; i < num_field_points; i++)
    localfield_data[i] = 0;

  for(large_numpart_type i = 0; i < num_on_grid; i += 8)
    {
      int fact          = pw_fold_factor(mode);
      int pindex        = (part[i].partindex >> 3);
      auto [dx, dy, dz] = cell_coordinates(Sp->get_IntPosition(pindex), fact);

      double weight = Sp->get_mass(pindex);

      if(mode) /* only for power spectrum calculation */
        continue;

      localfield_data[part[i + 0].localindex] += weight * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
      localfield_data[part[i + 1].localindex] += weight * (1.0 - dx) * (1.0 - dy) * dz;
      localfield_data[part[i + 2].localindex] += weight * (1.0 - dx) * dy * (1.0 - dz);
      localfield_data[part[i + 3].localindex] += weight * (1.0 - dx) * dy * dz;
      localfield_data[part[i + 4].localindex] += weight * (dx) * (1.0 - dy) * (1.0 - dz);
      localfield_data[part[i + 5].localindex] += weight * (dx) * (1.0 - dy) * dz;
      localfield_data[part[i + 6].localindex] += weight * (dx)*dy * (1.0 - dz);
      localfield_data[part[i + 7].localindex] += weight * (dx)*dy * dz;
    }

  rhogrid.resize(maxfftsize);
  std::fill(rhogrid.begin(), rhogrid.end(), 0);

  /* exchange data and add contributions to the local mesh-path */
  MPI_Alltoall(Sndpm_count.data(), sizeof(size_t), MPI_BYTE, Rcvpm_count.data(), sizeof(size_t), MPI_BYTE, Communicator);

  for(level = 0; level < (1 << PTask); level++) /* note: for level=0, target is the same task */
    {
      recvTask = ThisTask ^ level;
      large_array_offset *import_globalindex;
      fft_real *import_data;

      std::vector<fft_real> import_data_buf;
      std::vector<large_array_offset> import_globalindex_buf;

      if(recvTask < NTask)
        {
          if(level > 0)
            {
              import_data_buf.resize(Rcvpm_count[recvTask]);
              import_globalindex_buf.resize(Rcvpm_count[recvTask]);

              import_data        = import_data_buf.data();
              import_globalindex = import_globalindex_buf.data();

              if(Sndpm_count[recvTask] > 0 || Rcvpm_count[recvTask] > 0)
                {
                  myMPI_Sendrecv(localfield_data.data() + Sndpm_offset[recvTask], Sndpm_count[recvTask] * sizeof(fft_real), MPI_BYTE,
                                 recvTask, TAG_NONPERIOD_A, import_data, Rcvpm_count[recvTask] * sizeof(fft_real), MPI_BYTE, recvTask,
                                 TAG_NONPERIOD_A, Communicator, &status);

                  myMPI_Sendrecv(localfield_globalindex.data() + Sndpm_offset[recvTask],
                                 Sndpm_count[recvTask] * sizeof(large_array_offset), MPI_BYTE, recvTask, TAG_NONPERIOD_B,
                                 import_globalindex, Rcvpm_count[recvTask] * sizeof(large_array_offset), MPI_BYTE, recvTask,
                                 TAG_NONPERIOD_B, Communicator, &status);
                }
            }
          else
            {
              import_data        = localfield_data.data() + Sndpm_offset[ThisTask];
              import_globalindex = localfield_globalindex.data() + Sndpm_offset[ThisTask];
            }

          /* note: here every element in rhogrid is only accessed once, so there should be no race condition */
          for(size_t i = 0; i < Rcvpm_count[recvTask]; i++)
            {
              /* determine offset in local FFT slab */
#ifndef FFT_COLUMN_BASED
              large_array_offset offset =
                  import_globalindex[i] - first_slab_x_of_task[ThisTask] * Ngrid[1] * ((large_array_offset)Ngrid2);
#else
              large_array_offset offset = import_globalindex[i] - firstcol_XY * ((large_array_offset)Ngrid2);
#endif
              rhogrid[offset] += import_data[i];
            }
        }
    }
}

/* Function to read out the force component corresponding to spatial dimension 'dim'.
 * If dim is negative, potential values are read out and assigned to particles.
 */
void pm_periodic::pmforce_zoom_optimized_readout_forces_or_potential(fft_real *grid, int dim, const std::vector<part_slab_data> &part,
                                                                     std::vector<large_array_offset> &localfield_globalindex,
                                                                     std::vector<fft_real> &localfield_data,
                                                                     std::vector<std::array<double, 3>> &GravPM)
{
  // particle_data *P = Sp->P;

#ifdef EVALPOTENTIAL
#ifdef GRAVITY_TALLBOX
  double fac = 1.0 / (((double)Ngrid[0]) * Ngrid[1] * Ngrid[2]); /* to get potential  */
#else
  double fac = 4.0 * M_PI * (LONG_X * LONG_Y * LONG_Z) / pow(BoxSize, 3); /* to get potential  */
#endif
#endif

  for(int level = 0; level < (1 << PTask); level++) /* note: for level=0, target is the same task */
    {
      int recvTask = ThisTask ^ level;
      large_array_offset *import_globalindex;
      fft_real *import_data;

      std::vector<fft_real> import_data_buf;
      std::vector<large_array_offset> import_globalindex_buf;

      if(recvTask < NTask)
        {
          if(level > 0)
            {
              import_data_buf.resize(Rcvpm_count[recvTask]);
              import_globalindex_buf.resize(Rcvpm_count[recvTask]);

              import_data        = import_data_buf.data();
              import_globalindex = import_globalindex_buf.data();

              if(Sndpm_count[recvTask] > 0 || Rcvpm_count[recvTask] > 0)
                {
                  MPI_Status status;
                  myMPI_Sendrecv(localfield_globalindex.data() + Sndpm_offset[recvTask],
                                 Sndpm_count[recvTask] * sizeof(large_array_offset), MPI_BYTE, recvTask, TAG_NONPERIOD_C,
                                 import_globalindex, Rcvpm_count[recvTask] * sizeof(large_array_offset), MPI_BYTE, recvTask,
                                 TAG_NONPERIOD_C, Communicator, &status);
                }
            }
          else
            {
              import_data        = localfield_data.data() + Sndpm_offset[ThisTask];
              import_globalindex = localfield_globalindex.data() + Sndpm_offset[ThisTask];
            }

          for(size_t i = 0; i < Rcvpm_count[recvTask]; i++)
            {
#ifndef FFT_COLUMN_BASED
              large_array_offset offset =
                  import_globalindex[i] - first_slab_x_of_task[ThisTask] * Ngrid[1] * ((large_array_offset)Ngrid2);
#else
              large_array_offset offset = import_globalindex[i] - firstcol_XY * ((large_array_offset)Ngrid2);
#endif
              import_data[i] = grid[offset];
            }

          if(level > 0)
            {
              MPI_Status status;
              myMPI_Sendrecv(import_data, Rcvpm_count[recvTask] * sizeof(fft_real), MPI_BYTE, recvTask, TAG_NONPERIOD_A,
                             localfield_data.data() + Sndpm_offset[recvTask], Sndpm_count[recvTask] * sizeof(fft_real), MPI_BYTE,
                             recvTask, TAG_NONPERIOD_A, Communicator, &status);
            }
        }
    }

  /* read out the force/potential values, which all have been assembled in localfield_data */
  const int NSource = Sp->size();
  for(int idx = 0; idx < NSource; idx++)
    {
      // int i = Sp->get_active_index(idx);

      // #if !defined(HIERARCHICAL_GRAVITY) && defined(TREEPM_NOTIMESPLIT)
      //       if(!Sp->TimeBinSynchronized[P[i].TimeBinGrav])
      //         continue;
      // #endif

      large_numpart_type j = idx * 8;

      auto [dx, dy, dz] = cell_coordinates(Sp->get_IntPosition(idx));

      double value = localfield_data[part[j + 0].localindex] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) +
                     localfield_data[part[j + 1].localindex] * (1.0 - dx) * (1.0 - dy) * dz +
                     localfield_data[part[j + 2].localindex] * (1.0 - dx) * dy * (1.0 - dz) +
                     localfield_data[part[j + 3].localindex] * (1.0 - dx) * dy * dz +
                     localfield_data[part[j + 4].localindex] * (dx) * (1.0 - dy) * (1.0 - dz) +
                     localfield_data[part[j + 5].localindex] * (dx) * (1.0 - dy) * dz +
                     localfield_data[part[j + 6].localindex] * (dx)*dy * (1.0 - dz) +
                     localfield_data[part[j + 7].localindex] * (dx)*dy * dz;

      if(dim < 0)
        {
          // #ifdef EVALPOTENTIAL
          // #if defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
          //           P[i].PM_Potential += value * fac;
          // #else
          //           P[i].Potential += value * fac;
          // #endif
          // #endif
        }
      else
        {
          GravPM[idx][dim] += value;
        }
    }
}

#else

/*
 *  Here come the routines for a different communication algorithm that is better suited for a homogeneously loaded boxes.
 */
#ifndef FFT_COLUMN_BASED
void pm_periodic::pmforce_uniform_optimized_slabs_prepare_density(int mode, int *typelist, std::vector<partbuf> &partin)
{
  std::vector<partbuf> partout;
  // particle_data *P = Sp->P;

  /* determine the slabs/columns each particles accesses */

  for(int rep = 0; rep < 2; rep++)
    {
      /* each threads needs to do the loop to clear its send_count[] array */
      for(int j = 0; j < NTask; j++)
        Sndpm_count[j] = 0;

      const int NSource = Sp->size();
      for(int idx = 0; idx < NSource; idx++)
        {
          // int i = Sp->get_active_index(idx);

          if(mode) /* only for power spectrum calculation */
            continue;

          int fact = pw_fold_factor(mode);
          auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx), fact);
          int slab_xx = (slab_x + 1) % Ngrid[0];

          if(rep == 0)
            {
              int task0 = slab_to_task[slab_x];
              int task1 = slab_to_task[slab_xx];

              Sndpm_count[task0]++;

              if(task0 != task1)
                Sndpm_count[task1]++;
            }
          else
            {
              int task0 = slab_to_task[slab_x];
              int task1 = slab_to_task[slab_xx];

              size_t ind0 = Sndpm_offset[task0] + Sndpm_count[task0]++;
              partout[ind0].Mass = Sp->get_mass(idx);
              partout[ind0].IntPos = Sp->get_IntPosition(idx);

              if(task0 != task1)
                {
                  size_t ind1 = Sndpm_offset[task1] + Sndpm_count[task1]++;
                  partout[ind1].Mass = Sp->get_mass(idx);
                  partout[ind1].IntPos = Sp->get_IntPosition(idx);
                }
            }
        }

      if(rep == 0)
        {
          MPI_Alltoall(Sndpm_count.data(), sizeof(size_t), MPI_BYTE, Rcvpm_count.data(), sizeof(size_t), MPI_BYTE, Communicator);

          size_t nimport = 0, nexport = 0;
          Rcvpm_offset[0] = 0, Sndpm_offset[0] = 0;
          for(int j = 0; j < NTask; j++)
            {
              nexport += Sndpm_count[j];
              nimport += Rcvpm_count[j];

              if(j > 0)
                {
                  Sndpm_offset[j] = Sndpm_offset[j - 1] + Sndpm_count[j - 1];
                  Rcvpm_offset[j] = Rcvpm_offset[j - 1] + Rcvpm_count[j - 1];
                }
            }

          /* allocate import and export buffer */
          partin.resize(nimport);
          partout.resize(nexport);
        }
    }

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  int flag_big = 0, flag_big_all;
  for(int i = 0; i < NTask; i++)
    if(Sndpm_count[i] * sizeof(partbuf) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange particle data */
  myMPI_Alltoallv(partout.data(), Sndpm_count.data(), Sndpm_offset.data(), partin.data(), Rcvpm_count.data(), Rcvpm_offset.data(),
                  sizeof(partbuf), flag_big_all, Communicator);

  /* allocate cleared density field */
  rhogrid.resize(maxfftsize);
  std::fill(rhogrid.begin(), rhogrid.end(), 0);

  /* bin particle data onto mesh, in multi-threaded fashion */

  for(size_t i = 0; i < partin.size(); i++)
    {
      int fact = pw_fold_factor(mode);
      auto [slab_x, slab_y, slab_z] = grid_coordinates(partin[i].IntPos, fact);
      auto [dx, dy, dz] = cell_coordinates(partin[i].IntPos, fact);

      int slab_xx = (slab_x + 1) % Ngrid[0];
      int slab_yy = (slab_y + 1) % Ngrid[1];
      int slab_zz = (slab_z + 1) % Ngrid[2];

      double mass = partin[i].Mass;

      if(slab_to_task[slab_x] == ThisTask)
        {
          slab_x -= first_slab_x_of_task[ThisTask];

          rhogrid[FI(slab_x, slab_y, slab_z)] += (mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz));
          rhogrid[FI(slab_x, slab_y, slab_zz)] += (mass * (1.0 - dx) * (1.0 - dy) * (dz));

          rhogrid[FI(slab_x, slab_yy, slab_z)] += (mass * (1.0 - dx) * (dy) * (1.0 - dz));
          rhogrid[FI(slab_x, slab_yy, slab_zz)] += (mass * (1.0 - dx) * (dy) * (dz));
        }

      if(slab_to_task[slab_xx] == ThisTask)
        {
          slab_xx -= first_slab_x_of_task[ThisTask];

          rhogrid[FI(slab_xx, slab_y, slab_z)] += (mass * (dx) * (1.0 - dy) * (1.0 - dz));
          rhogrid[FI(slab_xx, slab_y, slab_zz)] += (mass * (dx) * (1.0 - dy) * (dz));

          rhogrid[FI(slab_xx, slab_yy, slab_z)] += (mass * (dx) * (dy) * (1.0 - dz));
          rhogrid[FI(slab_xx, slab_yy, slab_zz)] += (mass * (dx) * (dy) * (dz));
        }
    }
}
#else
void pm_periodic::pmforce_uniform_optimized_columns_prepare_density(int mode, int *typelist, std::vector<partbuf> &partin)
{
  std::vector<partbuf> partout;
  // particle_data *P = Sp->P;

  /* determine the slabs/columns each particles accesses */

  int columns         = Ngrid[0] * Ngrid[1];
  int avg             = (columns - 1) / NTask + 1;
  int exc             = NTask * avg - columns;
  int tasklastsection = NTask - exc;
  int pivotcol        = tasklastsection * avg;

  for(int rep = 0; rep < 2; rep++)
    {
      /* each threads needs to do the loop to clear its send_count[] array */
      for(int j = 0; j < NTask; j++)
        Sndpm_count[j] = 0;

      const int NSource = Sp->size();
      for(int idx = 0; idx < NSource; idx++)
        {
          // int i = Sp->get_active_index(idx);

          if(mode) /* only for power spectrum calculation */
            continue;

          int fact                      = pw_fold_factor(mode);
          auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx), fact);

          int slab_xx = slab_x + 1;

          if(slab_xx >= Ngrid[0])
            slab_xx = 0;

          int slab_yy = slab_y + 1;

          if(slab_yy >= Ngrid[1])
            slab_yy = 0;

          int column0 = slab_x * Ngrid[1] + slab_y;
          int column1 = slab_x * Ngrid[1] + slab_yy;
          int column2 = slab_xx * Ngrid[1] + slab_y;
          int column3 = slab_xx * Ngrid[1] + slab_yy;

          int task0, task1, task2, task3;

          if(column0 < pivotcol)
            task0 = column0 / avg;
          else
            task0 = (column0 - pivotcol) / (avg - 1) + tasklastsection;

          if(column1 < pivotcol)
            task1 = column1 / avg;
          else
            task1 = (column1 - pivotcol) / (avg - 1) + tasklastsection;

          if(column2 < pivotcol)
            task2 = column2 / avg;
          else
            task2 = (column2 - pivotcol) / (avg - 1) + tasklastsection;

          if(column3 < pivotcol)
            task3 = column3 / avg;
          else
            task3 = (column3 - pivotcol) / (avg - 1) + tasklastsection;

          if(rep == 0)
            {
              Sndpm_count[task0]++;
              if(task1 != task0)
                Sndpm_count[task1]++;
              if(task2 != task1 && task2 != task0)
                Sndpm_count[task2]++;
              if(task3 != task0 && task3 != task1 && task3 != task2)
                Sndpm_count[task3]++;
            }
          else
            {
              size_t ind0          = Sndpm_offset[task0] + Sndpm_count[task0]++;
              partout[ind0].Mass   = Sp->get_mass(idx);
              partout[ind0].IntPos = Sp->get_IntPosition(idx);

              if(task1 != task0)
                {
                  size_t ind1          = Sndpm_offset[task1] + Sndpm_count[task1]++;
                  partout[ind1].Mass   = Sp->get_mass(idx);
                  partout[ind1].IntPos = Sp->get_IntPosition(idx);
                }
              if(task2 != task1 && task2 != task0)
                {
                  size_t ind2          = Sndpm_offset[task2] + Sndpm_count[task2]++;
                  partout[ind2].Mass   = Sp->get_mass(idx);
                  partout[ind2].IntPos = Sp->get_IntPosition(idx);
                }
              if(task3 != task0 && task3 != task1 && task3 != task2)
                {
                  size_t ind3          = Sndpm_offset[task3] + Sndpm_count[task3]++;
                  partout[ind3].Mass   = Sp->get_mass(idx);
                  partout[ind3].IntPos = Sp->get_IntPosition(idx);
                }
            }
        }

      if(rep == 0)
        {
          MPI_Alltoall(Sndpm_count.data(), sizeof(size_t), MPI_BYTE, Rcvpm_count.data(), sizeof(size_t), MPI_BYTE, Communicator);

          size_t nimport = 0, nexport = 0;
          Rcvpm_offset[0] = 0, Sndpm_offset[0] = 0;
          for(int j = 0; j < NTask; j++)
            {
              nexport += Sndpm_count[j];
              nimport += Rcvpm_count[j];

              if(j > 0)
                {
                  Sndpm_offset[j] = Sndpm_offset[j - 1] + Sndpm_count[j - 1];
                  Rcvpm_offset[j] = Rcvpm_offset[j - 1] + Rcvpm_count[j - 1];
                }
            }

          /* allocate import and export buffer */
          partin.resize(nimport);
          partout.resize(nexport);
        }
    }

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  int flag_big = 0, flag_big_all;
  for(int i = 0; i < NTask; i++)
    if(Sndpm_count[i] * sizeof(partbuf) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange particle data */
  myMPI_Alltoallv(partout.data(), Sndpm_count.data(), Sndpm_offset.data(), partin.data(), Rcvpm_count.data(), Rcvpm_offset.data(),
                  sizeof(partbuf), flag_big_all, Communicator);

  /* allocate cleared density field */
  rhogrid.resize(maxfftsize);
  std::fill(rhogrid.begin(), rhogrid.end(), 0);

  int first_col = firstcol_XY;
  int last_col  = firstcol_XY + ncol_XY - 1;

  for(size_t i = 0; i < partin.size(); i++)
    {
      int fact                      = pw_fold_factor(mode);
      auto [slab_x, slab_y, slab_z] = grid_coordinates(partin[i].IntPos, fact);
      auto [dx, dy, dz]             = cell_coordinates(partin[i].IntPos, fact);

      int slab_xx = slab_x + 1;
      int slab_yy = slab_y + 1;

      if(slab_xx >= Ngrid[0])
        slab_xx = 0;

      if(slab_yy >= Ngrid[1])
        slab_yy = 0;

      int col0 = slab_x * Ngrid[1] + slab_y;
      int col1 = slab_x * Ngrid[1] + slab_yy;
      int col2 = slab_xx * Ngrid[1] + slab_y;
      int col3 = slab_xx * Ngrid[1] + slab_yy;

      double mass = partin[i].Mass;

      int slab_zz = slab_z + 1;

      if(slab_zz >= Ngrid[2])
        slab_zz = 0;

      if(col0 >= first_col && col0 <= last_col)
        {
          rhogrid[FCxy(col0, slab_z)] += (mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz));
          rhogrid[FCxy(col0, slab_zz)] += (mass * (1.0 - dx) * (1.0 - dy) * (dz));
        }

      if(col1 >= first_col && col1 <= last_col)
        {
          rhogrid[FCxy(col1, slab_z)] += (mass * (1.0 - dx) * (dy) * (1.0 - dz));
          rhogrid[FCxy(col1, slab_zz)] += (mass * (1.0 - dx) * (dy) * (dz));
        }

      if(col2 >= first_col && col2 <= last_col)
        {
          rhogrid[FCxy(col2, slab_z)] += (mass * (dx) * (1.0 - dy) * (1.0 - dz));
          rhogrid[FCxy(col2, slab_zz)] += (mass * (dx) * (1.0 - dy) * (dz));
        }

      if(col3 >= first_col && col3 <= last_col)
        {
          rhogrid[FCxy(col3, slab_z)] += (mass * (dx) * (dy) * (1.0 - dz));
          rhogrid[FCxy(col3, slab_zz)] += (mass * (dx) * (dy) * (dz));
        }
    }
}
#endif

/* If dim<0, this function reads out the potential, otherwise Cartesian force components.
 */
void pm_periodic::pmforce_uniform_optimized_readout_forces_or_potential_xy(fft_real *grid, int dim, const std::vector<partbuf> &partin,
                                                                           std::vector<std::array<double, 3>> &GravPM

)
{
  // particle_data *P = Sp->P;

#ifdef EVALPOTENTIAL
#ifdef GRAVITY_TALLBOX
  double fac = 1.0 / (((double)Ngrid[0]) * Ngrid[1] * Ngrid[2]); /* to get potential  */
#else
  double fac = 4 * M_PI * (LONG_X * LONG_Y * LONG_Z) / pow(BoxSize, 3); /* to get potential  */
#endif
#endif

  const size_t nexport = std::accumulate(Sndpm_count.begin(), Sndpm_count.end(), 0);
  std::vector<MyFloat> flistin(partin.size(), 0.0);
  std::vector<MyFloat> flistout(nexport);

#ifdef FFT_COLUMN_BASED
  int columns = Ngrid[0] * Ngrid[1];
  int avg = (columns - 1) / NTask + 1;
  int exc = NTask * avg - columns;
  int tasklastsection = NTask - exc;
  int pivotcol = tasklastsection * avg;
#endif  // FFT_COLUMN_BASED

  for(size_t i = 0; i < partin.size(); i++)
    {
      auto [slab_x, slab_y, slab_z] = grid_coordinates(partin[i].IntPos);
      auto [dx, dy, dz] = cell_coordinates(partin[i].IntPos);
      int slab_xx = (slab_x + 1) % Ngrid[0], slab_yy = (slab_y + 1) % Ngrid[1], slab_zz = (slab_z + 1) % Ngrid[2];

#ifndef FFT_COLUMN_BASED
      if(slab_to_task[slab_x] == ThisTask)
        {
          slab_x -= first_slab_x_of_task[ThisTask];

          flistin[i] += grid[FI(slab_x, slab_y, slab_z)] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) +
                        grid[FI(slab_x, slab_y, slab_zz)] * (1.0 - dx) * (1.0 - dy) * (dz) +
                        grid[FI(slab_x, slab_yy, slab_z)] * (1.0 - dx) * (dy) * (1.0 - dz) +
                        grid[FI(slab_x, slab_yy, slab_zz)] * (1.0 - dx) * (dy) * (dz);
        }

      if(slab_to_task[slab_xx] == ThisTask)
        {
          slab_xx -= first_slab_x_of_task[ThisTask];

          flistin[i] += grid[FI(slab_xx, slab_y, slab_z)] * (dx) * (1.0 - dy) * (1.0 - dz) +
                        grid[FI(slab_xx, slab_y, slab_zz)] * (dx) * (1.0 - dy) * (dz) +
                        grid[FI(slab_xx, slab_yy, slab_z)] * (dx) * (dy) * (1.0 - dz) +
                        grid[FI(slab_xx, slab_yy, slab_zz)] * (dx) * (dy) * (dz);
        }
#else   // yes FFT_COLUMN_BASED
      int column0 = slab_x * Ngrid[1] + slab_y;
      int column1 = slab_x * Ngrid[1] + slab_yy;
      int column2 = slab_xx * Ngrid[1] + slab_y;
      int column3 = slab_xx * Ngrid[1] + slab_yy;

      if(column0 >= firstcol_XY && column0 <= lastcol_XY)
        {
          flistin[i] += grid[FCxy(column0, slab_z)] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) +
                        grid[FCxy(column0, slab_zz)] * (1.0 - dx) * (1.0 - dy) * (dz);
        }
      if(column1 >= firstcol_XY && column1 <= lastcol_XY)
        {
          flistin[i] +=
              grid[FCxy(column1, slab_z)] * (1.0 - dx) * (dy) * (1.0 - dz) + grid[FCxy(column1, slab_zz)] * (1.0 - dx) * (dy) * (dz);
        }

      if(column2 >= firstcol_XY && column2 <= lastcol_XY)
        {
          flistin[i] +=
              grid[FCxy(column2, slab_z)] * (dx) * (1.0 - dy) * (1.0 - dz) + grid[FCxy(column2, slab_zz)] * (dx) * (1.0 - dy) * (dz);
        }

      if(column3 >= firstcol_XY && column3 <= lastcol_XY)
        {
          flistin[i] += grid[FCxy(column3, slab_z)] * (dx) * (dy) * (1.0 - dz) + grid[FCxy(column3, slab_zz)] * (dx) * (dy) * (dz);
        }
#endif  // FFT_COLUMN_BASED
    }

  /* exchange the potential component data */
  int flag_big = 0, flag_big_all;
  for(int i = 0; i < NTask; i++)
    if(Sndpm_count[i] * sizeof(MyFloat) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange  data */
  myMPI_Alltoallv(flistin.data(), Rcvpm_count.data(), Rcvpm_offset.data(), flistout.data(), Sndpm_count.data(), Sndpm_offset.data(),
                  sizeof(MyFloat), flag_big_all, Communicator);

  /* each threads needs to do the loop to clear its send_count[] array */
  for(int j = 0; j < NTask; j++)
    Sndpm_count[j] = 0;

  const int NSource = Sp->size();
  for(int idx = 0; idx < NSource; idx++)
    {
      // int i = Sp->get_active_index(idx);

      auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx));
      int slab_xx = slab_x + 1;

      if(slab_xx >= Ngrid[0])
        slab_xx = 0;

#ifndef FFT_COLUMN_BASED
      int task0 = slab_to_task[slab_x];
      int task1 = slab_to_task[slab_xx];

      double value = flistout[Sndpm_offset[task0] + Sndpm_count[task0]++];

      if(task0 != task1)
        value += flistout[Sndpm_offset[task1] + Sndpm_count[task1]++];
#else   // yes FFT_COLUMN_BASED
      int slab_yy = slab_y + 1;

      if(slab_yy >= Ngrid[1])
        slab_yy = 0;

      int column0 = slab_x * Ngrid[1] + slab_y;
      int column1 = slab_x * Ngrid[1] + slab_yy;
      int column2 = slab_xx * Ngrid[1] + slab_y;
      int column3 = slab_xx * Ngrid[1] + slab_yy;

      int task0, task1, task2, task3;

      if(column0 < pivotcol)
        task0 = column0 / avg;
      else
        task0 = (column0 - pivotcol) / (avg - 1) + tasklastsection;

      if(column1 < pivotcol)
        task1 = column1 / avg;
      else
        task1 = (column1 - pivotcol) / (avg - 1) + tasklastsection;

      if(column2 < pivotcol)
        task2 = column2 / avg;
      else
        task2 = (column2 - pivotcol) / (avg - 1) + tasklastsection;

      if(column3 < pivotcol)
        task3 = column3 / avg;
      else
        task3 = (column3 - pivotcol) / (avg - 1) + tasklastsection;

      double value = flistout[Sndpm_offset[task0] + Sndpm_count[task0]++];

      if(task1 != task0)
        value += flistout[Sndpm_offset[task1] + Sndpm_count[task1]++];

      if(task2 != task1 && task2 != task0)
        value += flistout[Sndpm_offset[task2] + Sndpm_count[task2]++];

      if(task3 != task0 && task3 != task1 && task3 != task2)
        value += flistout[Sndpm_offset[task3] + Sndpm_count[task3]++];
#endif  // FFT_COLUMN_BASED

      // #if !defined(HIERARCHICAL_GRAVITY) && defined(TREEPM_NOTIMESPLIT)
      //       if(!Sp->TimeBinSynchronized[Sp->P[i].TimeBinGrav])
      //         continue;
      // #endif

      if(dim < 0)
        {
          // #ifdef EVALPOTENTIAL
          // #if defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
          //           Sp->P[i].PM_Potential += value * fac;
          // #else
          //           Sp->P[i].Potential += value * fac;
          // #endif
          // #endif
        }
      else
        {
          GravPM[idx][dim] += value;
        }
    }
}

#ifdef FFT_COLUMN_BASED
void pm_periodic::pmforce_uniform_optimized_readout_forces_or_potential_xz(fft_real *grid, int dim,

                                                                           std::vector<std::array<double, 3>> &GravPM)
{
  if(dim != 1)
    Terminate("bummer");

  std::vector<size_t> send_count(NTask);
  std::vector<size_t> send_offset(NTask);
  std::vector<size_t> recv_count(NTask);
  std::vector<size_t> recv_offset(NTask);

  std::vector<partbuf> partin, partout;

  // particle_data *P = Sp->P;

  int columns = Ngrid[0] * Ngrid2;
  int avg = (columns - 1) / NTask + 1;
  int exc = NTask * avg - columns;
  int tasklastsection = NTask - exc;
  int pivotcol = tasklastsection * avg;

  /* determine the slabs/columns each particles accesses */
  for(int rep = 0; rep < 2; rep++)
    {
      for(int j = 0; j < NTask; j++)
        send_count[j] = 0;

      const int NSource = Sp->size();
      for(int idx = 0; idx < NSource; idx++)
        {
          // int i = Sp->get_active_index(idx);

          auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx));
          int slab_xx = slab_x + 1;

          if(slab_xx >= Ngrid[0])
            slab_xx = 0;

          int slab_zz = slab_z + 1;

          if(slab_zz >= Ngrid[2])
            slab_zz = 0;

          int column0 = slab_x * Ngrid2 + slab_z;
          int column1 = slab_x * Ngrid2 + slab_zz;
          int column2 = slab_xx * Ngrid2 + slab_z;
          int column3 = slab_xx * Ngrid2 + slab_zz;

          int task0, task1, task2, task3;

          if(column0 < pivotcol)
            task0 = column0 / avg;
          else
            task0 = (column0 - pivotcol) / (avg - 1) + tasklastsection;

          if(column1 < pivotcol)
            task1 = column1 / avg;
          else
            task1 = (column1 - pivotcol) / (avg - 1) + tasklastsection;

          if(column2 < pivotcol)
            task2 = column2 / avg;
          else
            task2 = (column2 - pivotcol) / (avg - 1) + tasklastsection;

          if(column3 < pivotcol)
            task3 = column3 / avg;
          else
            task3 = (column3 - pivotcol) / (avg - 1) + tasklastsection;

          if(rep == 0)
            {
              send_count[task0]++;
              if(task1 != task0)
                send_count[task1]++;
              if(task2 != task1 && task2 != task0)
                send_count[task2]++;
              if(task3 != task0 && task3 != task1 && task3 != task2)
                send_count[task3]++;
            }
          else
            {
              size_t ind0 = send_offset[task0] + send_count[task0]++;
              partout[ind0].IntPos = Sp->get_IntPosition(idx);

              if(task1 != task0)
                {
                  size_t ind1 = send_offset[task1] + send_count[task1]++;
                  partout[ind1].IntPos = Sp->get_IntPosition(idx);
                }
              if(task2 != task1 && task2 != task0)
                {
                  size_t ind2 = send_offset[task2] + send_count[task2]++;
                  partout[ind2].IntPos = Sp->get_IntPosition(idx);
                }
              if(task3 != task0 && task3 != task1 && task3 != task2)
                {
                  size_t ind3 = send_offset[task3] + send_count[task3]++;
                  partout[ind3].IntPos = Sp->get_IntPosition(idx);
                }
            }
        }

      if(rep == 0)
        {
          MPI_Alltoall(send_count.data(), sizeof(size_t), MPI_BYTE, recv_count.data(), sizeof(size_t), MPI_BYTE, Communicator);

          size_t nimport = 0, nexport = 0;
          recv_offset[0] = send_offset[0] = 0;

          for(int j = 0; j < NTask; j++)
            {
              nexport += send_count[j];
              nimport += recv_count[j];

              if(j > 0)
                {
                  send_offset[j] = send_offset[j - 1] + send_count[j - 1];
                  recv_offset[j] = recv_offset[j - 1] + recv_count[j - 1];
                }
            }

          /* allocate import and export buffer */
          partin.resize(nimport);
          partout.resize(nexport);
        }
    }

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  int flag_big = 0, flag_big_all;
  for(int i = 0; i < NTask; i++)
    if(send_count[i] * sizeof(partbuf) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange particle data */
  myMPI_Alltoallv(partout.data(), send_count.data(), send_offset.data(), partin.data(), recv_count.data(), recv_offset.data(),
                  sizeof(partbuf), flag_big_all, Communicator);

  std::vector<MyFloat> flistin(partin.size(), 0.0);
  std::vector<MyFloat> flistout(partout.size());

  for(size_t i = 0; i < partin.size(); i++)
    {
      auto [slab_x, slab_y, slab_z] = grid_coordinates(partin[i].IntPos);
      auto [dx, dy, dz] = cell_coordinates(partin[i].IntPos);
      int slab_xx = (slab_x + 1) % Ngrid[0], slab_yy = (slab_y + 1) % Ngrid[1], slab_zz = (slab_z + 1) % Ngrid[2];

      int column0 = slab_x * Ngrid2 + slab_z;
      int column1 = slab_x * Ngrid2 + slab_zz;
      int column2 = slab_xx * Ngrid2 + slab_z;
      int column3 = slab_xx * Ngrid2 + slab_zz;

      if(column0 >= firstcol_XZ && column0 <= lastcol_XZ)
        {
          flistin[i] += grid[FCxz(column0, slab_y)] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) +
                        grid[FCxz(column0, slab_yy)] * (1.0 - dx) * (dy) * (1.0 - dz);
        }
      if(column1 >= firstcol_XZ && column1 <= lastcol_XZ)
        {
          flistin[i] +=
              grid[FCxz(column1, slab_y)] * (1.0 - dx) * (1.0 - dy) * (dz) + grid[FCxz(column1, slab_yy)] * (1.0 - dx) * (dy) * (dz);
        }

      if(column2 >= firstcol_XZ && column2 <= lastcol_XZ)
        {
          flistin[i] +=
              grid[FCxz(column2, slab_y)] * (dx) * (1.0 - dy) * (1.0 - dz) + grid[FCxz(column2, slab_yy)] * (dx) * (dy) * (1.0 - dz);
        }

      if(column3 >= firstcol_XZ && column3 <= lastcol_XZ)
        {
          flistin[i] += grid[FCxz(column3, slab_y)] * (dx) * (1.0 - dy) * (dz) + grid[FCxz(column3, slab_yy)] * (dx) * (dy) * (dz);
        }
    }

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  flag_big = 0;
  for(int i = 0; i < NTask; i++)
    if(send_count[i] * sizeof(MyFloat) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange data */
  myMPI_Alltoallv(flistin.data(), recv_count.data(), recv_offset.data(), flistout.data(), send_count.data(), send_offset.data(),
                  sizeof(MyFloat), flag_big_all, Communicator);

  for(int j = 0; j < NTask; j++)
    send_count[j] = 0;

  /* now assign to original points */
  const int NSource = Sp->size();
  for(int idx = 0; idx < NSource; idx++)
    {
      // int i = Sp->get_active_index(idx);

      auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx));
      int slab_xx = slab_x + 1;

      if(slab_xx >= Ngrid[0])
        slab_xx = 0;

      int slab_zz = slab_z + 1;

      if(slab_zz >= Ngrid[2])
        slab_zz = 0;

      int column0 = slab_x * Ngrid2 + slab_z;
      int column1 = slab_x * Ngrid2 + slab_zz;
      int column2 = slab_xx * Ngrid2 + slab_z;
      int column3 = slab_xx * Ngrid2 + slab_zz;

      int task0, task1, task2, task3;

      if(column0 < pivotcol)
        task0 = column0 / avg;
      else
        task0 = (column0 - pivotcol) / (avg - 1) + tasklastsection;

      if(column1 < pivotcol)
        task1 = column1 / avg;
      else
        task1 = (column1 - pivotcol) / (avg - 1) + tasklastsection;

      if(column2 < pivotcol)
        task2 = column2 / avg;
      else
        task2 = (column2 - pivotcol) / (avg - 1) + tasklastsection;

      if(column3 < pivotcol)
        task3 = column3 / avg;
      else
        task3 = (column3 - pivotcol) / (avg - 1) + tasklastsection;

      double value = flistout[send_offset[task0] + send_count[task0]++];

      if(task1 != task0)
        value += flistout[send_offset[task1] + send_count[task1]++];

      if(task2 != task1 && task2 != task0)
        value += flistout[send_offset[task2] + send_count[task2]++];

      if(task3 != task0 && task3 != task1 && task3 != task2)
        value += flistout[send_offset[task3] + send_count[task3]++];

      // #if !defined(HIERARCHICAL_GRAVITY) && defined(TREEPM_NOTIMESPLIT)
      //       if(!Sp->TimeBinSynchronized[Sp->P[i].TimeBinGrav])
      //         continue;
      // #endif

      GravPM[idx][dim] += value;
    }
}

void pm_periodic::pmforce_uniform_optimized_readout_forces_or_potential_zy(fft_real *grid, int dim,
                                                                           std::vector<std::array<double, 3>> &GravPM

)
{
  if(dim != 0)
    Terminate("bummer");

  std::vector<size_t> send_count(NTask);
  std::vector<size_t> send_offset(NTask);
  std::vector<size_t> recv_count(NTask);
  std::vector<size_t> recv_offset(NTask);

  std::vector<partbuf> partin, partout;
  // defined in the header

  // particle_data *P = Sp->P;

  int columns = Ngrid2 * Ngrid[1];
  int avg = (columns - 1) / NTask + 1;
  int exc = NTask * avg - columns;
  int tasklastsection = NTask - exc;
  int pivotcol = tasklastsection * avg;

  /* determine the slabs/columns each particles accesses */
  for(int rep = 0; rep < 2; rep++)
    {
      for(int j = 0; j < NTask; j++)
        send_count[j] = 0;

      const int NSource = Sp->size();
      for(int idx = 0; idx < NSource; idx++)
        {
          // int i = Sp->get_active_index(idx);
          auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx));

          int slab_zz = slab_z + 1;

          if(slab_zz >= Ngrid[2])
            slab_zz = 0;

          int slab_yy = slab_y + 1;

          if(slab_yy >= Ngrid[1])
            slab_yy = 0;

          int column0 = slab_z * Ngrid[1] + slab_y;
          int column1 = slab_z * Ngrid[1] + slab_yy;
          int column2 = slab_zz * Ngrid[1] + slab_y;
          int column3 = slab_zz * Ngrid[1] + slab_yy;

          int task0, task1, task2, task3;

          if(column0 < pivotcol)
            task0 = column0 / avg;
          else
            task0 = (column0 - pivotcol) / (avg - 1) + tasklastsection;

          if(column1 < pivotcol)
            task1 = column1 / avg;
          else
            task1 = (column1 - pivotcol) / (avg - 1) + tasklastsection;

          if(column2 < pivotcol)
            task2 = column2 / avg;
          else
            task2 = (column2 - pivotcol) / (avg - 1) + tasklastsection;

          if(column3 < pivotcol)
            task3 = column3 / avg;
          else
            task3 = (column3 - pivotcol) / (avg - 1) + tasklastsection;

          if(rep == 0)
            {
              send_count[task0]++;
              if(task1 != task0)
                send_count[task1]++;
              if(task2 != task1 && task2 != task0)
                send_count[task2]++;
              if(task3 != task0 && task3 != task1 && task3 != task2)
                send_count[task3]++;
            }
          else
            {
              size_t ind0 = send_offset[task0] + send_count[task0]++;
              partout[ind0].IntPos = Sp->get_IntPosition(idx);

              if(task1 != task0)
                {
                  size_t ind1 = send_offset[task1] + send_count[task1]++;
                  partout[ind1].IntPos = Sp->get_IntPosition(idx);
                }
              if(task2 != task1 && task2 != task0)
                {
                  size_t ind2 = send_offset[task2] + send_count[task2]++;
                  partout[ind2].IntPos = Sp->get_IntPosition(idx);
                }
              if(task3 != task0 && task3 != task1 && task3 != task2)
                {
                  size_t ind3 = send_offset[task3] + send_count[task3]++;
                  partout[ind3].IntPos = Sp->get_IntPosition(idx);
                }
            }
        }

      if(rep == 0)
        {
          MPI_Alltoall(send_count.data(), sizeof(size_t), MPI_BYTE, recv_count.data(), sizeof(size_t), MPI_BYTE, Communicator);

          size_t nimport = 0, nexport = 0;
          recv_offset[0] = send_offset[0] = 0;

          for(int j = 0; j < NTask; j++)
            {
              nexport += send_count[j];
              nimport += recv_count[j];

              if(j > 0)
                {
                  send_offset[j] = send_offset[j - 1] + send_count[j - 1];
                  recv_offset[j] = recv_offset[j - 1] + recv_count[j - 1];
                }
            }

          /* allocate import and export buffer */
          partin.resize(nimport);
          partout.resize(nexport);
        }
    }

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  int flag_big = 0, flag_big_all;
  for(int i = 0; i < NTask; i++)
    if(send_count[i] * sizeof(partbuf) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange particle data */
  myMPI_Alltoallv(partout.data(), send_count.data(), send_offset.data(), partin.data(), recv_count.data(), recv_offset.data(),
                  sizeof(partbuf), flag_big_all, Communicator);

  std::vector<MyFloat> flistin(partin.size(), 0.0);
  std::vector<MyFloat> flistout(partout.size());

  for(size_t i = 0; i < partin.size(); i++)
    {
      auto [slab_x, slab_y, slab_z] = grid_coordinates(partin[i].IntPos);
      auto [dx, dy, dz] = cell_coordinates(partin[i].IntPos);
      int slab_xx = (slab_x + 1) % Ngrid[0], slab_yy = (slab_y + 1) % Ngrid[1], slab_zz = (slab_z + 1) % Ngrid[2];

      int column0 = slab_z * Ngrid[1] + slab_y;
      int column1 = slab_z * Ngrid[1] + slab_yy;
      int column2 = slab_zz * Ngrid[1] + slab_y;
      int column3 = slab_zz * Ngrid[1] + slab_yy;

      if(column0 >= firstcol_ZY && column0 <= lastcol_ZY)
        {
          flistin[i] += grid[FCzy(column0, slab_x)] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) +
                        grid[FCzy(column0, slab_xx)] * (dx) * (1.0 - dy) * (1.0 - dz);
        }
      if(column1 >= firstcol_ZY && column1 <= lastcol_ZY)
        {
          flistin[i] +=
              grid[FCzy(column1, slab_x)] * (1.0 - dx) * (dy) * (1.0 - dz) + grid[FCzy(column1, slab_xx)] * (dx) * (dy) * (1.0 - dz);
        }

      if(column2 >= firstcol_ZY && column2 <= lastcol_ZY)
        {
          flistin[i] +=
              grid[FCzy(column2, slab_x)] * (1.0 - dx) * (1.0 - dy) * (dz) + grid[FCzy(column2, slab_xx)] * (dx) * (1.0 - dy) * (dz);
        }

      if(column3 >= firstcol_ZY && column3 <= lastcol_ZY)
        {
          flistin[i] += grid[FCzy(column3, slab_x)] * (1.0 - dx) * (dy) * (dz) + grid[FCzy(column3, slab_xx)] * (dx) * (dy) * (dz);
        }
    }

  /* produce a flag if any of the send sizes is above our transfer limit, in this case we will
   * transfer the data in chunks.
   */
  flag_big = 0;
  for(int i = 0; i < NTask; i++)
    if(send_count[i] * sizeof(MyFloat) > MPI_MESSAGE_SIZELIMIT_IN_BYTES)
      flag_big = 1;

  MPI_Allreduce(&flag_big, &flag_big_all, 1, MPI_INT, MPI_MAX, Communicator);

  /* exchange data */
  myMPI_Alltoallv(flistin.data(), recv_count.data(), recv_offset.data(), flistout.data(), send_count.data(), send_offset.data(),
                  sizeof(MyFloat), flag_big_all, Communicator);

  for(int j = 0; j < NTask; j++)
    send_count[j] = 0;

  /* now assign to original points */
  const int NSource = Sp->size();
  for(int idx = 0; idx < NSource; idx++)
    {
      // int i = Sp->get_active_index(idx);
      auto [slab_x, slab_y, slab_z] = grid_coordinates(Sp->get_IntPosition(idx));

      int slab_zz = slab_z + 1;

      if(slab_zz >= Ngrid[2])
        slab_zz = 0;

      int slab_yy = slab_y + 1;

      if(slab_yy >= Ngrid[1])
        slab_yy = 0;

      int column0 = slab_z * Ngrid[1] + slab_y;
      int column1 = slab_z * Ngrid[1] + slab_yy;
      int column2 = slab_zz * Ngrid[1] + slab_y;
      int column3 = slab_zz * Ngrid[1] + slab_yy;

      int task0, task1, task2, task3;

      if(column0 < pivotcol)
        task0 = column0 / avg;
      else
        task0 = (column0 - pivotcol) / (avg - 1) + tasklastsection;

      if(column1 < pivotcol)
        task1 = column1 / avg;
      else
        task1 = (column1 - pivotcol) / (avg - 1) + tasklastsection;

      if(column2 < pivotcol)
        task2 = column2 / avg;
      else
        task2 = (column2 - pivotcol) / (avg - 1) + tasklastsection;

      if(column3 < pivotcol)
        task3 = column3 / avg;
      else
        task3 = (column3 - pivotcol) / (avg - 1) + tasklastsection;

      double value = flistout[send_offset[task0] + send_count[task0]++];

      if(task1 != task0)
        value += flistout[send_offset[task1] + send_count[task1]++];

      if(task2 != task1 && task2 != task0)
        value += flistout[send_offset[task2] + send_count[task2]++];

      if(task3 != task0 && task3 != task1 && task3 != task2)
        value += flistout[send_offset[task3] + send_count[task3]++];

      // #if !defined(HIERARCHICAL_GRAVITY) && defined(TREEPM_NOTIMESPLIT)
      //       if(!Sp->TimeBinSynchronized[Sp->P[i].TimeBinGrav])
      //         continue;
      // #endif

      GravPM[idx][dim] += value;
    }
}
#endif

#endif

/*! Calculates the long-range periodic force given the particle positions
 *  using the PM method.  The force is Gaussian filtered with Asmth, given in
 *  mesh-cell units. We carry out a CIC charge assignment, and compute the
 *  potential by fast Fourier transform methods. The potential is finite-differenced
 *  using a 4-point finite differencing formula, and the forces are
 *  interpolated tri-linearly to the particle positions. The CIC kernel is
 *  deconvolved.
 *
 *  For mode=0, normal force calculation, mode=1, only density field construction
 *  for a power spectrum calculation. In the later case, typelist flags the particle
 *  types that should be included in the density field.
 */
void pm_periodic::pmforce_periodic(int mode, int *typelist, double a)
{
  my_log << "calling " << __PRETTY_FUNCTION__ << "\n"; 
  std::vector<std::array<double, 3>> GravPM(Sp->size(), {0, 0, 0});

  // #ifdef HIERARCHICAL_GRAVITY
  //   NSource = Sp->TimeBinsGravity.NActiveParticles;
  // #else
  //   NSource = Sp->size();
  // #endif
  //
  // #ifndef TREEPM_NOTIMESPLIT
  //   if(NSource != Sp->size())
  //     Terminate("unexpected NSource != Sp->NumPart");
  // #endif

#ifndef NUMPART_PER_TASK_LARGE
  if((((long long)Sp->size()) << 3) >= (((long long)1) << 31))
    Terminate("We are dealing with a too large particle number per MPI rank - enabling NUMPART_PER_TASK_LARGE might help.");
#endif

#ifdef PM_ZOOM_OPTIMIZED
  std::vector<part_slab_data> part; /*!< array of part_slab_data linking the local particles to their mesh cells */
  std::vector<large_array_offset> localfield_globalindex;
  std::vector<fft_real> localfield_data;
  pmforce_zoom_optimized_prepare_density(mode, typelist, part, localfield_globalindex, localfield_data);
#else  // PM_ZOOM_OPTIMIZED
  std::vector<partbuf> partin;

#ifdef FFT_COLUMN_BASED
  pmforce_uniform_optimized_columns_prepare_density(mode, typelist, partin);
#else
  pmforce_uniform_optimized_slabs_prepare_density(mode, typelist, partin);
#endif  // FFT_COLUMN_BASED
#endif  // PM_ZOOM_OPTIMIZED

  /* note: after density, we still keep the field 'partin' from the density assignment,
   * as we can use this later on to return potential and z-force
   */

  /* allocate the memory to hold the FFT fields */

  forcegrid.resize(maxfftsize);
  auto &workspace = forcegrid;

  /* Do the FFT of the density field */
  fft(rhogrid.data(), workspace.data(), 1);

  if(mode != 0)
    {
      pmforce_measure_powerspec(mode - 1, typelist);
      return;
    }
  /* multiply with Green's function in order to obtain the potential (or forces for spectral diffencing) */
  compute_potential_kspace();

  /* Do the inverse FFT to get the potential/forces */

#ifndef FFT_COLUMN_BASED
  fft(rhogrid.data(), workspace.data(), -1);
#else   // FFT_COLUMN_BASED
  fft(workspace.data(), rhogrid.data(), -1);
#endif  // FFT_COLUMN_BASED

  /* Now rhogrid holds the potential/forces */

#ifdef EVALPOTENTIAL
#ifdef PM_ZOOM_OPTIMIZED
  pmforce_zoom_optimized_readout_forces_or_potential(rhogrid.data(), -1, part, localfield_globalindex, localfield_data, GravPM);
#else
  pmforce_uniform_optimized_readout_forces_or_potential_xy(rhogrid.data(), -1, partin, GravPM);
#endif
#endif

  /* get the force components by finite differencing of the potential for each dimension,
   * and send the results back to the right CPUs
   */

  /* we do the x component last, because for differencing the potential in the x-direction, we need to construct the
   * transpose
   */
#ifdef GRAVITY_TALLBOX
  double fac = 1.0 / (((double)Ngrid[0]) * Ngrid[1] * Ngrid[2]); /* to get potential  */
#else                                                            // GRAVITY_TALLBOX
  double fac = 4 * M_PI * (LONG_X * LONG_Y * LONG_Z) / pow(BoxSize, 3); /* to get potential  */
#endif                                                           // GRAVITY_TALLBOX

  const double d = BoxSize / Ngrid[0];
  fac *= 1 / (2 * d); /* for finite differencing */

#ifndef FFT_COLUMN_BASED

  /* z-direction */
  for(int y = 0; y < Ngrid[1]; y++)
    for(int x = 0; x < nslab_x; x++)
      for(int z = 0; z < Ngrid[2]; z++)
        {
          int zr = z + 1, zl = z - 1, zrr = z + 2, zll = z - 2;
          if(zr >= Ngrid[2])
            zr -= Ngrid[2];
          if(zrr >= Ngrid[2])
            zrr -= Ngrid[2];
          if(zl < 0)
            zl += Ngrid[2];
          if(zll < 0)
            zll += Ngrid[2];

          forcegrid[FI(x, y, z)] = fac * ((4.0 / 3) * (rhogrid[FI(x, y, zl)] - rhogrid[FI(x, y, zr)]) -
                                          (1.0 / 6) * (rhogrid[FI(x, y, zll)] - rhogrid[FI(x, y, zrr)]));
        }

#ifdef PM_ZOOM_OPTIMIZED
  pmforce_zoom_optimized_readout_forces_or_potential(forcegrid.data(), 2, part, localfield_globalindex, localfield_data, GravPM);
#else   // PM_ZOOM_OPTIMIZED
  pmforce_uniform_optimized_readout_forces_or_potential_xy(forcegrid.data(), 2, partin, GravPM);
#endif  // PM_ZOOM_OPTIMIZED

  /* y-direction */
  for(int y = 0; y < Ngrid[1]; y++)
    for(int x = 0; x < nslab_x; x++)
      for(int z = 0; z < Ngrid[2]; z++)
        {
          int yr = y + 1, yl = y - 1, yrr = y + 2, yll = y - 2;
          if(yr >= Ngrid[1])
            yr -= Ngrid[1];
          if(yrr >= Ngrid[1])
            yrr -= Ngrid[1];
          if(yl < 0)
            yl += Ngrid[1];
          if(yll < 0)
            yll += Ngrid[1];

          forcegrid[FI(x, y, z)] = fac * ((4.0 / 3) * (rhogrid[FI(x, yl, z)] - rhogrid[FI(x, yr, z)]) -
                                          (1.0 / 6) * (rhogrid[FI(x, yll, z)] - rhogrid[FI(x, yrr, z)]));
        }

#ifdef PM_ZOOM_OPTIMIZED
  pmforce_zoom_optimized_readout_forces_or_potential(forcegrid.data(), 1, part, localfield_globalindex, localfield_data, GravPM);
#else   // PM_ZOOM_OPTIMIZED
  pmforce_uniform_optimized_readout_forces_or_potential_xy(forcegrid.data(), 1, partin, GravPM);
#endif  // PM_ZOOM_OPTIMIZED

  /* x-direction */
  transposeA(rhogrid.data(), forcegrid.data()); /* compute the transpose of the potential field for finite differencing */
                                                /* note: for the x-direction, we difference the transposed field */

  for(int x = 0; x < Ngrid[0]; x++)
    for(int y = 0; y < nslab_y; y++)
      for(int z = 0; z < Ngrid[2]; z++)
        {
          int xrr = x + 2, xll = x - 2, xr = x + 1, xl = x - 1;
          if(xr >= Ngrid[0])
            xr -= Ngrid[0];
          if(xrr >= Ngrid[0])
            xrr -= Ngrid[0];
          if(xl < 0)
            xl += Ngrid[0];
          if(xll < 0)
            xll += Ngrid[0];

          forcegrid[NI(x, y, z)] = fac * ((4.0 / 3) * (rhogrid[NI(xl, y, z)] - rhogrid[NI(xr, y, z)]) -
                                          (1.0 / 6) * (rhogrid[NI(xll, y, z)] - rhogrid[NI(xrr, y, z)]));
        }

  transposeB(forcegrid.data(), rhogrid.data()); /* reverse the transpose from above */

#ifdef PM_ZOOM_OPTIMIZED
  pmforce_zoom_optimized_readout_forces_or_potential(forcegrid.data(), 0, part, localfield_globalindex, localfield_data, GravPM);
#else   // PM_ZOOM_OPTIMIZED
  pmforce_uniform_optimized_readout_forces_or_potential_xy(forcegrid.data(), 0, partin, GravPM);
#endif  // PM_ZOOM_OPTIMIZED

#else  // yes FFT_COLUMN_BASED

  /* z-direction */
  for(large_array_offset i = 0; i < ncol_XY; i++)
    {
      fft_real *const forcep = &forcegrid[Ngrid2 * i];
      fft_real *const potp = &rhogrid[Ngrid2 * i];

      for(int z = 0; z < Ngrid[2]; z++)
        {
          int zr = z + 1;
          int zl = z - 1;
          int zrr = z + 2;
          int zll = z - 2;

          if(zr >= Ngrid[2])
            zr -= Ngrid[2];
          if(zrr >= Ngrid[2])
            zrr -= Ngrid[2];
          if(zl < 0)
            zl += Ngrid[2];
          if(zll < 0)
            zll += Ngrid[2];

          forcep[z] = fac * ((4.0 / 3) * (potp[zl] - potp[zr]) - (1.0 / 6) * (potp[zll] - potp[zrr]));
        }
    }

#ifdef PM_ZOOM_OPTIMIZED
  pmforce_zoom_optimized_readout_forces_or_potential(forcegrid.data(), 2, part, localfield_globalindex, localfield_data, GravPM);
#else   // PM_ZOOM_OPTIMIZED
  pmforce_uniform_optimized_readout_forces_or_potential_xy(forcegrid.data(), 2, partin, GravPM);
#endif  // PM_ZOOM_OPTIMIZED

  /* y-direction */
  swap23(rhogrid.data(), forcegrid.data());  // rhogrid contains potential field, forcegrid the transposed field

  /* make an in-place computation */
  {
    std::vector<fft_real> column(Ngrid[1]);

    for(large_array_offset i = 0; i < ncol_XZ; i++)
      {
        std::memcpy(column.data(), &forcegrid[Ngrid[1] * i], Ngrid[1] * sizeof(fft_real));

        fft_real *const forcep = &forcegrid[Ngrid[1] * i];

        for(int y = 0; y < Ngrid[1]; y++)
          {
            int yr = y + 1;
            int yl = y - 1;
            int yrr = y + 2;
            int yll = y - 2;

            if(yr >= Ngrid[1])
              yr -= Ngrid[1];
            if(yrr >= Ngrid[1])
              yrr -= Ngrid[1];
            if(yl < 0)
              yl += Ngrid[1];
            if(yll < 0)
              yll += Ngrid[1];

            forcep[y] = fac * ((4.0 / 3) * (column[yl] - column[yr]) - (1.0 / 6) * (column[yll] - column[yrr]));
          }
      }
  }

  /* now need to read out from forcegrid  in a non-standard way */

#ifdef PM_ZOOM_OPTIMIZED
  /* need a third field as scratch space */
  {
    std::vector<fft_real> scratch(fftsize);
    swap23back(forcegrid.data(), scratch.data());
    pmforce_zoom_optimized_readout_forces_or_potential(scratch.data(), 1, part, localfield_globalindex, localfield_data, GravPM);
  }
#else   // PM_ZOOM_OPTIMIZED
  pmforce_uniform_optimized_readout_forces_or_potential_xz(forcegrid.data(), 1, GravPM);
#endif  // PM_ZOOM_OPTIMIZED

  /* x-direction */
  swap13(rhogrid.data(), forcegrid.data());  // rhogrid contains potential field

  for(large_array_offset i = 0; i < ncol_ZY; i++)
    {
      fft_real *forcep = &rhogrid[Ngrid[0] * i];
      fft_real *potp = &forcegrid[Ngrid[0] * i];

      for(int x = 0; x < Ngrid[0]; x++)
        {
          int xr = x + 1;
          int xl = x - 1;
          int xrr = x + 2;
          int xll = x - 2;

          if(xr >= Ngrid[0])
            xr -= Ngrid[0];
          if(xrr >= Ngrid[0])
            xrr -= Ngrid[0];
          if(xl < 0)
            xl += Ngrid[0];
          if(xll < 0)
            xll += Ngrid[0];

          forcep[x] = fac * ((4.0 / 3) * (potp[xl] - potp[xr]) - (1.0 / 6) * (potp[xll] - potp[xrr]));
        }
    }

    /* now need to read out from forcegrid in a non-standard way */
#ifdef PM_ZOOM_OPTIMIZED
  swap13back(rhogrid.data(), forcegrid.data());
  pmforce_zoom_optimized_readout_forces_or_potential(forcegrid.data(), 0, part, localfield_globalindex, localfield_data, GravPM);
#else   // PM_ZOOM_OPTIMIZED
  pmforce_uniform_optimized_readout_forces_or_potential_zy(rhogrid.data(), 0, GravPM);
#endif  // PM_ZOOM_OPTIMIZED

#endif  // FFT_COLUMN_BASED

  // write GravPM to particles
  for(int i = 0; i < GravPM.size(); ++i)
    Sp->set_acceleration(i, GravPM[i]);
  
  // update the velocities
  const double inv_a = 1.0/a;
  for(int i = 0; i < Sp->size(); ++i)
  {
    auto p = Sp->get_momentum(i);
    for(auto &px : p)
        px *= inv_a;
    
    // TODO: velocity here
    // Sp->set_velocity(i,p);
  }
}

#ifdef GRAVITY_TALLBOX

/*! This function sets-up the Greens function for calculating the tall-box potential
 *  in real space, with suitable zero padding in the direction of the tall box.
 */
void pm_periodic::pmforce_setup_tallbox_kernel(void)
{
  double d = BoxSize / Ngrid[0];

  /* now set up kernel and its Fourier transform */

  for(int i = 0; i < maxfftsize; i++) /* clear local field */
    kernel[i] = 0;

#ifndef FFT_COLUMN_BASED
  for(int i = slabstart_x; i < (slabstart_x + nslab_x); i++)
    for(int j = 0; j < Ngrid[1]; j++)
      {
#else
  for(int c = firstcol_XY; c < (firstcol_XY + ncol_XY); c++)
    {
      int i = c / Ngrid[1];
      int j = c % Ngrid[1];
#endif

        for(int k = 0; k < Ngrid[2]; k++)
          {
            int ii, jj, kk;

            if(i >= (Ngrid[0] / 2))
              ii = i - Ngrid[0];
            else
              ii = i;
            if(j >= (Ngrid[1] / 2))
              jj = j - Ngrid[1];
            else
              jj = j;
            if(k >= (Ngrid[2] / 2))
              kk = k - Ngrid[2];
            else
              kk = k;

            double xx = ii * d;
            double yy = jj * d;
            double zz = kk * d;

            double pot = pmperiodic_tallbox_long_range_potential(xx, yy, zz);

#ifndef FFT_COLUMN_BASED
            size_t ip = FI(i - slabstart_x, j, k);
#else
          size_t ip = FCxy(c, k);
#endif
            kernel[ip] = pot / BoxSize;
          }

#ifndef FFT_COLUMN_BASED
      }
#else
    }
#endif

  /* Do the FFT of the kernel */

  std::vector<fft_real> workspc(maxfftsize);

  fft(kernel.get(), workspc.data(), 1); /* result is in workspace, not in kernel */
#ifndef FFT_COLUMN_BASED
#else
  std::memcpy(kernel.get(), workspc.data(), maxfftsize * sizeof(fft_real));
#endif
}

double pm_periodic::pmperiodic_tallbox_long_range_potential(double x, double y, double z)
{
  x /= BoxSize;
  y /= BoxSize;
  z /= BoxSize;

  double r = sqrt(x * x + y * y + z * z);

  if(r == 0)
    return 0;

  double xx, yy, zz;
  switch(GRAVITY_TALLBOX)
    {
      case 0:
        xx = y;
        yy = z;
        zz = x;
        break;
      case 1:
        xx = x;
        yy = z;
        zz = y;
        break;
      case 2:
        xx = x;
        yy = y;
        zz = z;
        break;
    }
  x = xx;
  y = yy;
  z = zz;

  /* the third dimension, z, is now the non-periodic one */

  double leff  = sqrt(BOXX * BOXY);
  double alpha = 2.0 / leff;

  double sum1 = 0.0;

  int qxmax = (int)(10.0 / (BOXX * alpha) + 0.5);
  int qymax = (int)(10.0 / (BOXY * alpha) + 0.5);

  int nxmax = (int)(4.0 * alpha * BOXX + 0.5);
  int nymax = (int)(4.0 * alpha * BOXY + 0.5);

  for(int nx = -qxmax; nx <= qxmax; nx++)
    for(int ny = -qymax; ny <= qymax; ny++)
      {
        double dx = x - nx * BOXX;
        double dy = y - ny * BOXY;
        double r  = sqrt(dx * dx + dy * dy + z * z);
        if(r > 0)
          sum1 += erfc(alpha * r) / r;
      }

  double alpha2 = alpha * alpha;

  double sum2 = 0.0;

  for(int nx = -nxmax; nx <= nxmax; nx++)
    for(int ny = -nymax; ny <= nymax; ny++)
      {
        if(nx != 0 || ny != 0)
          {
            double kx = (2.0 * M_PI / BOXX) * nx;
            double ky = (2.0 * M_PI / BOXY) * ny;
            double k2 = kx * kx + ky * ky;
            double k  = sqrt(k2);

            if(k * z > 0)
              {
                double ex = exp(-k * z);
                if(ex > 0)
                  sum2 += cos(kx * x + ky * y) * (erfc(k / (2 * alpha) + alpha * z) / ex + ex * erfc(k / (2 * alpha) - alpha * z)) / k;
              }
            else
              {
                double ex = exp(k * z);
                if(ex > 0)
                  sum2 += cos(kx * x + ky * y) * (ex * erfc(k / (2 * alpha) + alpha * z) + erfc(k / (2 * alpha) - alpha * z) / ex) / k;
              }
          }
      }

  sum2 *= M_PI / (BOXX * BOXY);

  double psi = 2.0 * alpha / sqrt(M_PI) +
               (2 * sqrt(M_PI) / (BOXX * BOXY) * (exp(-alpha2 * z * z) / alpha + sqrt(M_PI) * z * erf(alpha * z))) - (sum1 + sum2);

  return psi;
}
#endif

/*----------------------------------------------------------------------------------------------------*/
/*           Here comes code for the power-spectrum computation                                       */
/*----------------------------------------------------------------------------------------------------*/

void pm_periodic::calculate_power_spectra(int num, char *OutputDir)
{
  int n_type[NTYPES];
  long long ntot_type_all[NTYPES];
  /* determine global and local particle numbers */
  for(int n = 0; n < NTYPES; n++)
    n_type[n] = 0;
  const int NSource = Sp->size();
  for(int n = 0; n < NSource; n++)
    //   n_type[Sp->P[n].getType()]++;
    n_type[0]++;

  sumup_large_ints(NTYPES, n_type, ntot_type_all, Communicator);

  int typeflag[NTYPES];

  for(int i = 0; i < NTYPES; i++)
    typeflag[i] = 1;

#ifdef HIERARCHICAL_GRAVITY
  int flag_extra_allocate = 0;
  if(Sp->TimeBinsGravity.ActiveParticleList == NULL)
    {
      flag_extra_allocate = 1;
      Sp->TimeBinsGravity.timebins_allocate();
    }

  Sp->TimeBinsGravity.NActiveParticles = 0;
  for(int i = 0; i < NSource; i++)
    Sp->TimeBinsGravity.ActiveParticleList[Sp->TimeBinsGravity.NActiveParticles++] = i;
#endif

  if(ThisTask == 0)
    {
      char buf[MAXLEN_PATH_EXTRA];
      sprintf(buf, "%s/powerspecs", OutputDir);
      mkdir(buf, 02755);
    }

  sprintf(power_spec_fname, "%s/powerspecs/powerspec_%03d.txt", OutputDir, num);

  pmforce_do_powerspec(typeflag); /* calculate power spectrum for all particle types */

  /* check whether whether more than one type is in use */
  int count_types = 0;
  for(int i = 0; i < NTYPES; i++)
    if(ntot_type_all[i] > 0)
      count_types++;

  if(count_types > 1)
    for(int i = 0; i < NTYPES; i++)
      {
        if(ntot_type_all[i] > 0)
          {
            for(int j = 0; j < NTYPES; j++)
              typeflag[j] = 0;

            typeflag[i] = 1;

            sprintf(power_spec_fname, "%s/powerspecs/powerspec_type%d_%03d.txt", OutputDir, i, num);

            pmforce_do_powerspec(typeflag); /* calculate power spectrum for type i */
          }
      }

#ifdef HIERARCHICAL_GRAVITY
  if(flag_extra_allocate)
    Sp->TimeBinsGravity.timebins_free();
#endif
}

void pm_periodic::pmforce_do_powerspec(int *typeflag)
{
  mpi_printf("POWERSPEC: Begin power spectrum. (typeflag=[");
  for(int i = 0; i < NTYPES; i++)
    mpi_printf(" %d ", typeflag[i]);
  mpi_printf("])\n");

  double tstart = MPI_Wtime();

  pmforce_periodic(1, typeflag); /* calculate regular power spectrum for selected particle types */

  pmforce_periodic(2, typeflag); /* calculate folded power spectrum for selected particle types  */

  pmforce_periodic(3, typeflag); /* calculate twice folded power spectrum for selected particle types  */

  double tend = MPI_Wtime();

  mpi_printf("POWERSPEC: End power spectrum. took %g seconds\n", tend - tstart);
}

void pm_periodic::pmforce_measure_powerspec(int flag, int *typeflag)
{
  // particle_data *P = Sp->P;

  long long CountModes[BINS_PS];
  double SumPowerUncorrected[BINS_PS]; /* without binning correction (as for shot noise) */
  double PowerUncorrected[BINS_PS];    /* without binning correction */
  double DeltaUncorrected[BINS_PS];    /* without binning correction */
  double ShotLimit[BINS_PS];
  double KWeightSum[BINS_PS];
  double Kbin[BINS_PS];

  double mass = 0, mass2 = 0, count = 0;
  const int NSource = Sp->size();
  for(int i = 0; i < NSource; i++)
    // if(typeflag[P[i].getType()])
    {
      double m = Sp->get_mass(i);
      mass += m;
      mass2 += m * m;
      count += 1.0;
    }

  MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, Communicator);
  MPI_Allreduce(MPI_IN_PLACE, &mass2, 1, MPI_DOUBLE, MPI_SUM, Communicator);
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_DOUBLE, MPI_SUM, Communicator);

  double d     = BoxSize / Ngrid[0];
  double dhalf = 0.5 * d;

  double fac = 1.0 / mass;

  double K0     = 2 * M_PI / BoxSize;                                                          /* minimum k */
  double K1     = 2 * M_PI / BoxSize * (POWERSPEC_FOLDFAC * POWERSPEC_FOLDFAC * Ngrid[0] / 2); /* maximum k that can be measured */
  double binfac = BINS_PS / (log(K1) - log(K0));

  double kfacx = 2.0 * M_PI * LONG_X / BoxSize;
  double kfacy = 2.0 * M_PI * LONG_Y / BoxSize;
  double kfacz = 2.0 * M_PI * LONG_Z / BoxSize;

  for(int i = 0; i < BINS_PS; i++)
    {
      SumPowerUncorrected[i] = 0;
      CountModes[i]          = 0;
      KWeightSum[i]          = 0;
    }

#ifdef FFT_COLUMN_BASED
  for(large_array_offset ip = 0; ip < second_transposed_ncells; ip++)
    {
      large_array_offset ipcell = ip + ((large_array_offset)second_transposed_firstcol) * Ngrid[0];
      int y                     = ipcell / (Ngrid[0] * Ngridz);
      int yr                    = ipcell % (Ngrid[0] * Ngridz);
      int z                     = yr / Ngrid[0];
      int x                     = yr % Ngrid[0];
#else
  for(int y = slabstart_y; y < slabstart_y + nslab_y; y++)
    for(int x = 0; x < Ngrid[0]; x++)
      for(int z = 0; z < Ngridz; z++)
        {
#endif
      int count_double;

      if(z >= 1 &&
         z < (Ngrid[2] + 1) / 2) /* these modes need to be counted twice due to the storage scheme for the FFT of a real field */
        count_double = 1;
      else
        count_double = 0;

      int xx, yy, zz;

      if(x >= (Ngrid[0] / 2))
        xx = x - Ngrid[0];
      else
        xx = x;

      if(y >= (Ngrid[1] / 2))
        yy = y - Ngrid[1];
      else
        yy = y;

      if(z >= (Ngrid[2] / 2))
        zz = z - Ngrid[2];
      else
        zz = z;

      double kx = kfacx * xx;
      double ky = kfacy * yy;
      double kz = kfacz * zz;

      double k2 = kx * kx + ky * ky + kz * kz;

      if(k2 > 0)
        {
          /* do deconvolution */
          double fx = 1, fy = 1, fz = 1;

          if(xx != 0)
            {
              fx = kx * dhalf;
              fx = sin(fx) / fx;
            }
          if(yy != 0)
            {
              fy = ky * dhalf;
              fy = sin(fy) / fy;
            }
          if(zz != 0)
            {
              fz = kz * dhalf;
              fz = sin(fz) / fz;
            }
          double ff   = 1 / (fx * fy * fz);
          double smth = ff * ff * ff * ff;
          /*
           * Note: The Fourier-transform of the density field (rho_hat) must be multiplied with ff^2
           * in order to do the de-convolution. Thats why po = rho_hat^2 gains a factor of ff^4.
           */
          /* end deconvolution */

#ifndef FFT_COLUMN_BASED
          large_array_offset ip = ((large_array_offset)Ngridz) * (Ngrid[0] * (y - slabstart_y) + x) + z;
#endif

          const std::complex<fft_real> *const fft_of_rhogrid = (std::complex<fft_real> *)rhogrid.data();
          double po                                          = std::norm(fft_of_rhogrid[ip]);

          po *= fac * fac * smth;

          double k = sqrt(k2);

          if(flag == 1)
            k *= POWERSPEC_FOLDFAC;
          else if(flag == 2)
            k *= POWERSPEC_FOLDFAC * POWERSPEC_FOLDFAC;

          if(k >= K0 && k < K1)
            {
              int bin = log(k / K0) * binfac;

              SumPowerUncorrected[bin] += po;
              CountModes[bin] += 1;
              KWeightSum[bin] += log(k);

              if(count_double)
                {
                  SumPowerUncorrected[bin] += po;
                  CountModes[bin] += 1;
                  KWeightSum[bin] += log(k);
                }
            }
        }
    }

  MPI_Allreduce(MPI_IN_PLACE, SumPowerUncorrected, BINS_PS, MPI_DOUBLE, MPI_SUM, Communicator);
  MPI_Allreduce(MPI_IN_PLACE, CountModes, BINS_PS, MPI_LONG_LONG, MPI_SUM, Communicator);
  MPI_Allreduce(MPI_IN_PLACE, KWeightSum, BINS_PS, MPI_DOUBLE, MPI_SUM, Communicator);

  int count_non_zero_bins = 0;
  for(int i = 0; i < BINS_PS; i++)
    {
      if(CountModes[i] > 0)
        {
          Kbin[i] = exp(KWeightSum[i] / CountModes[i]);
          count_non_zero_bins++;
        }
      else
        Kbin[i] = exp((i + 0.5) / binfac + log(K0));

      if(CountModes[i] > 0)
        PowerUncorrected[i] = SumPowerUncorrected[i] / CountModes[i];
      else
        PowerUncorrected[i] = 0;

      DeltaUncorrected[i] = 4 * M_PI * pow(Kbin[i], 3) / pow(2 * M_PI / BoxSize, 3) * PowerUncorrected[i];

      ShotLimit[i] = 4 * M_PI * pow(Kbin[i], 3) / pow(2 * M_PI / BoxSize, 3) * (mass2 / (mass * mass));
    }

  /* store the result */
  if(ThisTask == 0)
    {
      FILE *fd;

      if(flag == 0)
        {
          if(!(fd = fopen(power_spec_fname, "w"))) /* store the unfolded spectrum */
            Terminate("can't open file `%s`\n", power_spec_fname);
        }
      else if(flag == 1 || flag == 2)
        {
          if(!(fd = fopen(power_spec_fname, "a"))) /* append the file, store the folded spectrum */
            Terminate("can't open file `%s`\n", power_spec_fname);
        }
      else
        Terminate("Something wrong.\n");

      // fprintf(fd, "%g\n", All.Time);
      fprintf(fd, "%d\n", count_non_zero_bins);
      fprintf(fd, "%g\n", BoxSize);
      fprintf(fd, "%d\n", (int)(Ngrid[0]));
      // if(All.ComovingIntegrationOn)
      //  fprintf(fd, "%g\n", All.ComovingIntegrationOn > 0 ? linear_growth_factor(All.Time, 1.0) : 1.0);

      for(int i = 0; i < BINS_PS; i++)
        if(CountModes[i] > 0)
          fprintf(fd, "%g %g %g %g %g\n", Kbin[i], DeltaUncorrected[i], PowerUncorrected[i], (double)CountModes[i], ShotLimit[i]);

      if(flag == 2)
        {
          fprintf(fd, "%g\n", mass);
          fprintf(fd, "%g\n", count);
          fprintf(fd, "%g\n", mass * mass / mass2);
        }

      fclose(fd);
    }
}

#ifdef FFT_COLUMN_BASED
/* multiply with Green's function in order to obtain the potential (or forces for spectral diffencing) */
void pm_periodic::compute_potential_kspace()
{
  std::complex<fft_real> *const fft_of_rhogrid = (std::complex<fft_real> *)forcegrid.data();

  for(large_array_offset ip = 0; ip < second_transposed_ncells; ip++)
    {
      int x, y, z;
      large_array_offset ipcell = ip + ((large_array_offset)second_transposed_firstcol) * Ngrid[0];
      y                         = ipcell / (Ngrid[0] * Ngridz);
      int yr                    = ipcell % (Ngrid[0] * Ngridz);
      z                         = yr / Ngrid[0];
      x                         = yr % Ngrid[0];
      const int xx = signed_mode(x, Ngrid[0]), yy = signed_mode(y, Ngrid[1]), zz = signed_mode(z, Ngrid[2]);

      fft_of_rhogrid[ip] *= green_function({xx, yy, zz});

#ifdef GRAVITY_TALLBOX
      fft_of_rhogrid[ip] *= fft_of_kernel[ip];
#endif
    }

#ifndef GRAVITY_TALLBOX
  if(second_transposed_firstcol == 0)
    fft_of_rhogrid[0] = 0.0;
#endif
}
#else
/* multiply with Green's function in order to obtain the potential (or forces for spectral diffencing) */
void pm_periodic::compute_potential_kspace()
{
  std::complex<fft_real> *const fft_of_rhogrid = (std::complex<fft_real> *)rhogrid.data();

  for(int x = 0; x < Ngrid[0]; x++)
    for(int y = slabstart_y; y < slabstart_y + nslab_y; y++)
      for(int z = 0; z < Ngridz; z++)
        {
          large_array_offset ip = ((large_array_offset)Ngridz) * (Ngrid[0] * (y - slabstart_y) + x) + z;
          const int xx = signed_mode(x, Ngrid[0]), yy = signed_mode(y, Ngrid[1]), zz = signed_mode(z, Ngrid[2]);

          fft_of_rhogrid[ip] *= green_function({xx, yy, zz});

#ifdef GRAVITY_TALLBOX
          fft_of_rhogrid[ip] *= fft_of_kernel[ip];
#endif
        }

#ifndef GRAVITY_TALLBOX
  if(slabstart_y == 0)
    fft_of_rhogrid[0] = 0.0;
#endif
}
#endif

double pm_periodic::green_function(std::array<int, 3> mode) const
{
  const double dhalf = 0.5 * BoxSize / Ngrid[0];
  std::array<double, 3> k;
  double k2{0.0};

  for(int i = 0; i < 3; ++i)
    k[i] = mode[i] * k_fundamental(i);
  for(auto kx : k)
    k2 += kx * kx;

  double smth = 1.0, deconv = 1.0;

  if(k2 > 0)
    {
      smth = -std::exp(-k2 * asmth2) / k2;

      /* do deconvolution */
      double ff = 1.0;

      for(int i = 0; i < 3; ++i)
        if(mode[i] != 0)
          {
            double fx = k[i] * dhalf;
            ff *= fx / std::sin(fx);
          }
      deconv = ff * ff * ff * ff;
      // deconv = ff * ff;  // CIC is 2nd order
      smth *= deconv; /* deconvolution */
    }

#ifdef GRAVITY_TALLBOX
  return deconv * std::exp(-k2 * asmth2);
#else
  return smth;
#endif
}


} // namespace gadget
