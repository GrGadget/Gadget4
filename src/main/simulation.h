/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  simulation.h
 *
 *  \brief declares the main simulation class holding its data and principle modules
 */

#ifndef SIMULATION_H
#define SIMULATION_H

#include "gadgetconfig.h"

#include <mpi.h>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/lcparticles.h"
#include "../data/mmparticles.h"
#include "../data/mymalloc.h"
#include "../data/simparticles.h"
#include "../domain/domain.h"
#include "../fmm/fmm.h"
#include "../fof/fof.h"
#include "../gravity/ewald.h"
#include "../gravity/grav_forcetest.h"
#include "../gravtree/gravtree.h"
#include "../gravtree/gwalk.h"
#include "../io/restart.h"
#include "../io/test_io_bandwidth.h"
#include "../lightcone/lightcone.h"
#include "../logs/logs.h"
#include "../mergertree/mergertree.h"
#include "../ngbtree/ngbtree.h"
#include "../ngenic/ngenic.h"
#include "../pm/pm.h"
#include "../sph/sph.h"
#include "gadget/constants.h"
#include "gadget/debugger.hpp"
#include "gadget/dtypes.h"
#include "gadget/macros.h"
#include "gadget/mpi_utils.h"
#include "gadget/parameters.h"
#include "gadget/setcomm.h"
#include "gadget/addons/latfield_handler.hpp"

#include <string>

namespace gadget{

#define GRID (HRPMGRID)

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

#endif

class sim : public test_io_bandwidth
{
 public:
  sim(MPI_Comm comm)
      : setcomm(comm),
        test_io_bandwidth(comm)
#ifdef PMGRID
#ifndef GEVOLUTION_PM
#ifdef PERIODIC
        ,
        PM(comm, {GRIDX, GRIDY, GRIDZ})
#else
        ,
        PM(comm, HRPMGRID)
#endif
#else
        ,
        PM(comm,GRIDX)
#endif
#endif
  {
  }

  /* here come the main classes the code operates on */

  simparticles Sp{Communicator}; /* stores all the simulation particles of the simulation */

  domain<simparticles> Domain{Communicator, &Sp}; /* get an instance of a domain decomposition, operating on Sp */

  /* Note: GravTree/NgbTree  inherit their communicator and NTask/ThisTask from their associated domain object  */

#ifdef FMM
  fmm GravTree; /* get an instance of a gravitational tree */
#else
  gwalk GravTree; /* get an instance of a gravitational tree */
#endif

  sph NgbTree; /* get an instance of a neighbour search tree */


#ifdef PMGRID
  pm PM;
#endif

#ifdef MERGERTREE
  mergertree MergerTree{Communicator, &Sp};
#endif

#ifdef COOLING
  coolsfr CoolSfr{Communicator};
#endif

#ifdef NGENIC
  ngenic Ngenic{Communicator, &Sp};
#endif

#ifdef LIGHTCONE
#ifdef LIGHTCONE_PARTICLES
  lcparticles Lp{Communicator}; /* stores all the buffered light-cone particles of the simulation */
#endif
#ifdef LIGHTCONE_MASSMAPS
  mmparticles Mp{Communicator}; /* stores buffered particles to construct projected mass maps on the lightcone */
#endif

#if defined(LIGHTCONE_PARTICLES) && defined(LIGHTCONE_MASSMAPS) /* both particles and massmaps */
  lightcone LightCone{Communicator, &Sp, &Lp, &Mp};
#else
#if defined(LIGHTCONE_PARTICLES)
  lightcone LightCone{Communicator, &Sp, &Lp};
#else
  lightcone LightCone{Communicator, &Sp, &Mp};
#endif
#endif
#endif  // end of LIGHTCONE

#ifdef FORCETEST
  gravtest GravTest{&Sp, &GravTree, &Domain};
#endif

  void rearrange_lightcone(int argc, char **argv);
  void rearrange_snapshot(int argc, char **argv);

  template <typename partset>
  void rearrange_generic(partset &Tp, int conenr, int firstnr, int lastnr);

  template <typename partset>
  void rearrange_fill_treetable(partset &Tp);

  template <typename partset>
  void rearrange_read(partset &Tp, int num, int conenr);

  template <typename partset>
  void rearrange_write(partset &Tp, int num, int conenr);

 private:
  
  #ifdef FORCETEST
  gadget::debugger_t treepm_forces{Communicator,"treepm_forces.bin"};
  gadget::debugger_t pm_forces{Communicator,"pm_forces.bin"};
  gadget::debugger_t summation_forces{Communicator,"summation_forces.bin"};
  #endif
 
  void sincronize_particles();
#ifdef PERIODIC
  void check_omega(void);
#endif

  void setup_smoothinglengths(void);
  void recreate_unique_ids(void);
  void test_id_uniqueness(void);
  int check_for_interruption_of_run(void);
  void set_non_standard_physics_for_current_time(void);
  void calculate_non_standard_physics_end_of_step(void);
  integertime find_next_outputtime(integertime ti_curr);
  double measure_cpu_performance(MPI_Comm Communicator);
  double measure_hyper_cube_speed(const char *tag, MPI_Comm Communicator);
  void measure_iprobe_performance(const char *tag);

#ifdef SECOND_ORDER_LPT_ICS
  double F1_Omega(double a);
  double F2_Omega(double a);
  int executed = 0;
  void second_order_ic_correction(void);
#endif

 public:
  void hello(void);
  void endrun(void);
  void begrun1(const char *parameterFile);
  void begrun2(void);
  void init(int RestartSnapNum);
  void run(void);
  void set_units(void);
  void create_snapshot_if_desired(void);
  void healthtest(void);
  void mpi_report_comittable_memory(void);
  long long report_comittable_memory(long long *MemTotal, long long *Committed_AS, long long *SwapTotal, long long *SwapFree);
  long long report_free_size_in_tmpfs(void);
  void do_gravity_step_second_half(void);
  void find_timesteps_and_do_gravity_step_first_half(void);
  void do_hydro_step_first_half(void);
  void do_hydro_step_second_half(void);
  void find_global_timesteps(void);
  void find_hydro_timesteps(void);
  void gravity(int timebin);
  void gravity_long_range_force(void);
  void gravity_comoving_factors(int timebin);
  void gravity_pm(int timebin);
  void gravity_set_oldacc(int timebin);
  void save_forces_to_file();

  void hydro_force(int step_indicator);
  void compute_grav_accelerations(int timebin);
#ifdef FORCETEST_TESTFORCELAW
  void gravity_forcetest_testforcelaw(void);
#endif

#ifdef EXTERNALGRAVITY
  void gravity_external(void);
#endif
#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  void init_individual_softenings(void);
  static int get_softening_type_from_mass(double mass);
  static double get_desired_softening_from_mass(double mass);
#endif
#ifdef ADAPTIVE_HYDRO_SOFTENING
  int get_softeningtype_for_hydro_particle(int i);
#endif
  void print_particle_info_from_ID(MyIDType ID);
  void print_particle_info(int i);
};

}
#endif
