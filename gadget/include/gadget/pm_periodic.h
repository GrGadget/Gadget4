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
#include <limits>  // numeric_limits
#include <memory>  // unique_ptr
#include <tuple>
#include <vector>
#include <sstream>
extern template class std::vector<size_t>;

#include "gadget/constants.h"         // MAXLEN_PATH_EXTRA
#include "gadget/dtypes.h"            // MyIntPosType
#include "gadget/particle_handler.h"  // particle_handler
#include "gadget/pm_mpi_fft.h"        // pm_mpi_fft
#include "gadgetconfig.h"

namespace gadget{

class pm_periodic :

#ifdef FFT_COLUMN_BASED
    public mpi_fft_columns
#else
    public mpi_fft_slabs
#endif
{
  std::stringstream my_log;
  const MyIntPosType INTCELL;
  double asmth2;

 public:
  std::string get_log()
  {
    std::string log_mes = my_log.str();
    my_log.str("");
    return log_mes;
  }
 
  pm_periodic(MPI_Comm comm, std::array<int, 3> ngrid)
      :
#ifdef FFT_COLUMN_BASED
        mpi_fft_columns(comm, ngrid),
        Sndpm_count(NTask),
        Sndpm_offset(NTask),
        Rcvpm_count(NTask),
        Rcvpm_offset(NTask)
#else
        mpi_fft_slabs(comm, ngrid),
        Sndpm_count(NTask),
        Sndpm_offset(NTask),
        Rcvpm_count(NTask),
        Rcvpm_offset(NTask)
#endif
        ,
        INTCELL{std::numeric_limits<MyIntPosType>::max() / Ngrid[0] + 1}
  {
  }
    
  int size()const{return Ngrid[0];}  
  void pm_init_periodic(particle_handler *Sp_ptr, double boxsize, double asmth);
  void pmforce_periodic(int mode, int *typelist, double a = 1);
  void calculate_power_spectra(int num, char *OutputDir);

 private:
  /* short-cut macros for accessing different 3D arrays */
  inline auto FI(int x, int y, int z) const { return ((large_array_offset)Ngrid2) * (Ngrid[1] * x + y) + z; }
#ifdef FFT_COLUMN_BASED
  inline auto FCxy(int c, int z) const { return ((large_array_offset)Ngrid2) * (c - firstcol_XY) + z; }
  inline auto FCxz(int c, int y) const { return ((large_array_offset)Ngrid[1]) * (c - firstcol_XZ) + y; }
  inline auto FCzy(int c, int x) const { return ((large_array_offset)Ngrid[0]) * (c - firstcol_ZY) + x; }
#else
  inline auto NI(int x, int y, int z) const { return ((large_array_offset)Ngrid[2]) * (y + x * nslab_y) + z; }
#endif

  typedef long long large_array_offset; /* use a larger data type in this case so that we can always address all cells of the 3D grid
                                           with a single index */
  std::vector<size_t> Sndpm_count, Sndpm_offset, Rcvpm_count, Rcvpm_offset;
  double BoxSize{};
  std::unique_ptr<gadget::particle_handler> Sp;
  char power_spec_fname[MAXLEN_PATH_EXTRA];

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
  static int pw_fold_factor(int mode) noexcept
  {
    int fact = 1;
    switch(mode)
      {
        case 2:
          fact = POWERSPEC_FOLDFAC;
          break;
        case 3:
          fact = POWERSPEC_FOLDFAC * POWERSPEC_FOLDFAC;
          break;
        default:
          fact = 1;
      }
    return fact;
  }
  double k_fundamental(int dim=0) const noexcept
  {
    double d = BoxSize;  // double d = Ngrid[dim] * BoxSize / PMGRID;
    return 2.0 * M_PI / d;
  }
  double green_function(std::array<int, 3> mode) const;

  // template <typename part_t>
  auto grid_coordinates(std::array<MyIntPosType,3> IntPos, int fold_fact = 1 /*for power
  spectrum computations only*/)const
  {
    std::array<int, 3> slab;
    for(int i = 0; i < 3; ++i)
      slab[i] = (IntPos[i] * fold_fact) / INTCELL;
    return slab;
  }

  // template <typename part_t>
  auto cell_coordinates(std::array<MyIntPosType, 3> IntPos, int fold_fact = 1 /*for power spectrum
     computations only*/) const
  {
    std::array<double, 3> dx;
    for(int i = 0; i < 3; ++i)
      {
        long long int rmd = (IntPos[i] * fold_fact) % INTCELL;
        dx[i]             = rmd * (1.0 / INTCELL);
      }
    return dx;
  }

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
                                                          std::vector<fft_real> &localfield_data,
                                                          std::vector<std::array<double, 3>> &GravPM);
#else

  struct partbuf
  {
#ifndef LEAN
    MyFloat Mass;
#else
    static MyFloat Mass;
#endif
    std::array<MyIntPosType, 3> IntPos;
  };

#ifndef FFT_COLUMN_BASED
  void pmforce_uniform_optimized_slabs_prepare_density(int mode, int *typelist, std::vector<partbuf> &partin);
#else
  void pmforce_uniform_optimized_columns_prepare_density(int mode, int *typelist, std::vector<partbuf> &partin);
#endif
  void pmforce_uniform_optimized_readout_forces_or_potential_xy(fft_real *grid, int dim, const std::vector<partbuf> &partin,
                                                                std::vector<std::array<double, 3>> &GravPM);
  void pmforce_uniform_optimized_readout_forces_or_potential_xz(fft_real *grid, int dim, std::vector<std::array<double, 3>> &GravPM);
  void pmforce_uniform_optimized_readout_forces_or_potential_zy(fft_real *grid, int dim, std::vector<std::array<double, 3>> &GravPM);
#endif
};

}
