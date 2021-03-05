/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  pm_periodic.h
 *
 *  \brief declaration of a class used for periodic PM-force calculations
 */

#pragma once

#include <array>
#include <complex>
#include <memory>  // unique_ptr
#include <tuple>
#include <vector>
extern template class std::vector<size_t>;

#include "../data/simparticles.h"  // simparticles
#include "gadget/constants.h"      // MAXLEN_PATH_EXTRA
#include "gadget/dtypes.h"         // LONG_X
#include "gadgetconfig.h"

#include "gadget/pm_mpi_fft.h"  // pm_mpi_fft

#if defined(PMGRID) && defined(PERIODIC)

#ifdef LONG_X_BITS
#if PMGRID != ((PMGRID / LONG_X) * LONG_X)
#error "PMGRID must be a multiple of the stretch factor in the x-direction"
#endif
#endif

#ifdef LONG_Y_BITS
#if PMGRID != ((PMGRID / LONG_Y) * LONG_Y)
#error "PMGRID must be a multiple of the stretch factor in the y-direction"
#endif
#endif

#ifdef LONG_Z_BITS
#if PMGRID != ((PMGRID / LONG_Z) * LONG_Z)
#error "PMGRID must be a multiple of the stretch factor in the x-direction"
#endif
#endif

#define GRIDX ((PMGRID / LONG_X) * DBX + DBX_EXTRA)
#define GRIDY ((PMGRID / LONG_Y) * DBY + DBY_EXTRA)
#define GRIDZ ((PMGRID / LONG_Z) * DBZ + DBZ_EXTRA)

#define INTCELL ((~((MyIntPosType)0)) / PMGRID + 1)
#endif

class pm_periodic :

#ifdef FFT_COLUMN_BASED
    public mpi_fft_columns
#else
    public mpi_fft_slabs
#endif
{
 public:
  pm_periodic(MPI_Comm comm)
#ifdef FFT_COLUMN_BASED
      : mpi_fft_columns(comm, GRIDX, GRIDY, GRIDZ), Sndpm_count(NTask), Sndpm_offset(NTask), Rcvpm_count(NTask), Rcvpm_offset(NTask)
#else
      : mpi_fft_slabs(comm, GRIDX, GRIDY, GRIDZ), Sndpm_count(NTask), Sndpm_offset(NTask), Rcvpm_count(NTask), Rcvpm_offset(NTask)
#endif
  {
  }

  void pm_init_periodic(simparticles *Sp_ptr, double boxsize);
  void pmforce_periodic(int mode, int *typelist);
  void calculate_power_spectra(int num, char *OutputDir);

 private:
  typedef long long large_array_offset; /* use a larger data type in this case so that we can always address all cells of the 3D grid
                                           with a single index */
  std::vector<size_t> Sndpm_count, Sndpm_offset, Rcvpm_count, Rcvpm_offset;
  double BoxSize{};
  simparticles *Sp;
  char power_spec_fname[MAXLEN_PATH_EXTRA];
  int NSource;

  /*! \var maxfftsize
   *  \brief maximum size of the local fft grid among all tasks
   */
  long long maxfftsize;

  /*! \var rhogrid
   *  \brief This array hold the local part of the density field and
   *  after the FFTs the local part of the potential
   *
   *  \var forcegrid
   *  \brief This array will contain the force field
   *
   *  \var workspace
   *  \brief Workspace array used during the FFTs
   */
  std::vector<fft_real> rhogrid, forcegrid;  //, *workspace;

  /* variables for power spectrum estimation */
  static constexpr int BINS_PS           = 4000; /* number of bins for power spectrum computation */
  static constexpr int POWERSPEC_FOLDFAC = 16;   /* folding factor to obtain an estimate of the power spectrum on very small scales */

  void pmforce_measure_powerspec(int flag, int *typeflag);
  void pmforce_do_powerspec(int *typeflag);
  void compute_potential_kspace();
  static int signed_mode(int x, int L) noexcept { return x >= L / 2 ? x - L : x; }
  int k_fundamental(int dim) const noexcept
  {
    double d = BoxSize / PMGRID;
    switch(dim)
      {
        case 0:
          d *= GRIDX;
          break;
        case 1:
          d *= GRIDY;
          break;
        case 2:
          d *= GRIDZ;
          break;
      }
    return 2.0 * M_PI / d;
  }
  double green_function(std::array<int, 3> mode) const;

#if defined(GRAVITY_TALLBOX)
  std::unique_ptr<fft_real[]> kernel; /*!< If the tallbox option is used, the code will construct and store the k-space Greens function
                       by FFTing it from real space */
  std::complex<fft_real> *fft_of_kernel;
  void pmforce_setup_tallbox_kernel(void);
  double pmperiodic_tallbox_long_range_potential(double x, double y, double z);
#endif

#ifdef PM_ZOOM_OPTIMIZED

  /*! \brief This structure links the particles to the mesh cells, to which they contribute their mass
   *
   * Each particle will have eight items of this structure in the #part array.
   * For each of the eight mesh cells the CIC assignment will contribute,
   * one item of this struct exists.
   */

  struct part_slab_data
  {
    large_array_offset globalindex; /*!< index in the global density mesh */
    large_numpart_type partindex;   /*!< contains the local particle index shifted by 2^3, the first three bits encode to which part of
                                       the CIC assignment this item belongs to */
    large_array_offset localindex;  /*!< index to a local copy of the corresponding mesh cell of the global density array (used during
                                       local mass and force assignment) */
  };

  void pmforce_zoom_optimized_prepare_density(int mode, int *typelist, std::vector<part_slab_data> &part,
                                              std::vector<large_array_offset> &localfield_globalindex,
                                              std::vector<fft_real> &localfield_data);
  void pmforce_zoom_optimized_readout_forces_or_potential(fft_real *grid, int dim, const std::vector<part_slab_data> &part,
                                                          std::vector<large_array_offset> &localfield_globalindex,
                                                          std::vector<fft_real> &localfield_data);
#else

  struct partbuf
  {
#ifndef LEAN
    MyFloat Mass;
#else
    static MyFloat Mass;
#endif
    MyIntPosType IntPos[3];
  };

  template <typename part_t>
  static auto coordinates(const part_t &P, int fold_fact = 1 /*for power spectrum computations only*/)
  {
    std::array<int, 3> slab;
    for(int i = 0; i < 3; ++i)
      slab[i] = (P.IntPos[i] * fold_fact) / INTCELL;
    return slab;
  }

  template <typename part_t>
  static auto cell_coordinates(const part_t &P, int fold_fact = 1 /*for power spectrum computations only*/)
  {
    std::array<double, 3> dx;
    for(int i = 0; i < 3; ++i)
      {
        MyIntPosType rmd = (P.IntPos[i] * fold_fact) % INTCELL;
        dx[i]            = rmd * (1.0 / INTCELL);
      }
    return dx;
  }

#ifndef FFT_COLUMN_BASED
  void pmforce_uniform_optimized_slabs_prepare_density(int mode, int *typelist, std::vector<partbuf> &partin);
#else
  void pmforce_uniform_optimized_columns_prepare_density(int mode, int *typelist, std::vector<partbuf> &partin);
#endif
  void pmforce_uniform_optimized_readout_forces_or_potential_xy(fft_real *grid, int dim, const std::vector<partbuf> &partin);
  void pmforce_uniform_optimized_readout_forces_or_potential_xz(fft_real *grid, int dim);
  void pmforce_uniform_optimized_readout_forces_or_potential_zy(fft_real *grid, int dim);
#endif
};
