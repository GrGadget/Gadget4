/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  timestep.cc
 *
 *  \brief routines for determining the timesteps of particles
 */

#include "gadgetconfig.h"

#include <string.h>  // strncpy
#include <cstdio>    // snprintf

#include "gadget/timebindata.h"

namespace gadget{ 
void TimeBinData::timebins_init(const char *name, const int MaxPart)
{
  NActiveParticles = 0;

  for(int i = 0; i < TIMEBINS; i++)
    {
      FirstInTimeBin[i] = -1;
      LastInTimeBin[i]  = -1;
    }

  strncpy(Name, name, 99);
  Name[99] = 0;
  timebins_reallocate(MaxPart);
}

void TimeBinData::timebins_reallocate(const int MaxPart)
{
  ActiveParticleList.resize(MaxPart);
  NextInTimeBin.resize(MaxPart);
  PrevInTimeBin.resize(MaxPart);
}

void TimeBinData::timebin_move_particle(int p, int timeBin_old, int timeBin_new)
{
  if(timeBin_old == timeBin_new)
    return;

  TimeBinCount[timeBin_old]--;

  int prev = PrevInTimeBin[p];
  int next = NextInTimeBin[p];

  if(FirstInTimeBin[timeBin_old] == p)
    FirstInTimeBin[timeBin_old] = next;
  if(LastInTimeBin[timeBin_old] == p)
    LastInTimeBin[timeBin_old] = prev;
  if(prev >= 0)
    NextInTimeBin[prev] = next;
  if(next >= 0)
    PrevInTimeBin[next] = prev;

  if(TimeBinCount[timeBin_new] > 0)
    {
      PrevInTimeBin[p]                          = LastInTimeBin[timeBin_new];
      NextInTimeBin[LastInTimeBin[timeBin_new]] = p;
      NextInTimeBin[p]                          = -1;
      LastInTimeBin[timeBin_new]                = p;
    }
  else
    {
      FirstInTimeBin[timeBin_new] = LastInTimeBin[timeBin_new] = p;
      PrevInTimeBin[p] = NextInTimeBin[p] = -1;
    }

  TimeBinCount[timeBin_new]++;
}

void TimeBinData::timebin_remove_particle(int idx, int bin)
{
  int p                   = ActiveParticleList[idx];
  ActiveParticleList[idx] = -1;

  TimeBinCount[bin]--;

  if(p >= 0)
    {
      int prev = PrevInTimeBin[p];
      int next = NextInTimeBin[p];

      if(prev >= 0)
        NextInTimeBin[prev] = next;
      if(next >= 0)
        PrevInTimeBin[next] = prev;

      if(FirstInTimeBin[bin] == p)
        FirstInTimeBin[bin] = next;
      if(LastInTimeBin[bin] == p)
        LastInTimeBin[bin] = prev;
    }
}

/* insert a particle into the timebin struct behind another already existing particle */
void TimeBinData::timebin_add_particle(int i_new, int i_old, int timeBin, int addToListOfActiveParticles)
{
  TimeBinCount[timeBin]++;

  if(i_old < 0)
    {
      /* if we don't have an existing particle to add if after, let's take the last one in this timebin */
      i_old = LastInTimeBin[timeBin];

      if(i_old < 0)
        {
          /* the timebin is empty at the moment, so just add the new particle */
          FirstInTimeBin[timeBin] = i_new;
          LastInTimeBin[timeBin]  = i_new;
          NextInTimeBin[i_new]    = -1;
          PrevInTimeBin[i_new]    = -1;
        }
    }

  if(i_old >= 0)
    {
      /* otherwise we added it already */
      PrevInTimeBin[i_new] = i_old;
      NextInTimeBin[i_new] = NextInTimeBin[i_old];
      if(NextInTimeBin[i_old] >= 0)
        PrevInTimeBin[NextInTimeBin[i_old]] = i_new;
      NextInTimeBin[i_old] = i_new;
      if(LastInTimeBin[timeBin] == i_old)
        LastInTimeBin[timeBin] = i_new;
    }

  if(addToListOfActiveParticles)
    {
      ActiveParticleList[NActiveParticles] = i_new;
      NActiveParticles++;
    }
}

void TimeBinData::timebin_make_list_of_active_particles_up_to_timebin(int timebin)
{
  NActiveParticles = 0;
  for(int tbin = timebin; tbin >= 0; tbin--)
    timebin_add_particles_of_timebin_to_list_of_active_particles(tbin);
}

void TimeBinData::timebin_add_particles_of_timebin_to_list_of_active_particles(int timebin)
{
  for(int i = FirstInTimeBin[timebin]; i >= 0; i = NextInTimeBin[i])
    {
      ActiveParticleList[NActiveParticles] = i;
      NActiveParticles++;
    }
}
}
