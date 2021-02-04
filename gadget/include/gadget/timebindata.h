/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  timestep.h
 *
 *  \brief declarations of a structure for organizing the timebins
 */

#ifndef TIMESTEP_H
#define TIMESTEP_H

#include <vector>
#include "gadget/dtypes.h"  // TIMEBINS
#include "gadgetconfig.h"

extern template class std::vector<int>;

struct TimeBinData
{
 public:
  int NActiveParticles;
  long long GlobalNActiveParticles;
  int TimeBinCount[TIMEBINS];

  int FirstInTimeBin[TIMEBINS];
  int LastInTimeBin[TIMEBINS];

  char Name[100];
  // int *MaxPart;

  std::vector<int> ActiveParticleList;
  std::vector<int> NextInTimeBin;
  std::vector<int> PrevInTimeBin;

  /* TimeBinData stuff */
  void timebins_init(const char *name, const int MaxPart);
  void timebins_reallocate(const int MaxPart);
  void timebin_move_particle(int p, int timeBin_old, int timeBin_new);
  void timebin_add_particle(int i_new, int i_old, int timeBin, int addToListOfActiveParticles);
  void timebin_remove_particle(int idx, int bin);
  void timebin_cleanup_list_of_active_particles(void);
  void timebin_move_sfr(int p, int timeBin_old, int timeBin_new);
  void timebin_move_bh(int p, int timeBin_old, int timeBin_new);
  void timebin_make_list_of_active_particles_up_to_timebin(int timebin);
  void timebin_add_particles_of_timebin_to_list_of_active_particles(int timebin);
};

#endif /* TIMESTEP */
