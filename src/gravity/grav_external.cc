/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file gravity_external.cc
 *
 * \brief can add an optional external gravity field
 */

#include "gadgetconfig.h"

#ifdef EXTERNALGRAVITY

#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "../data/allvars.h"
#include "../data/mymalloc.h"
#include "../domain/domain.h"
#include "../fmm/fmm.h"
#include "../gravtree/gravtree.h"
#include "../logs/logs.h"
#include "../main/simulation.h"
#include "../system/system.h"
#include "gadget/dtypes.h"
#include "gadget/intposconvert.h"
#include "gadget/mpi_utils.h"
#include "gadget/timebindata.h"

void sim::gravity_external(void)
{
  for(int i = 0; i < Sp.TimeBinsGravity.NActiveParticles; i++)
    {
      int target = Sp.TimeBinsGravity.ActiveParticleList[i];

#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
      Sp.P[target].ExtPotential = 0;
#endif

#ifdef EXTERNALGRAVITY_STATICHQ
      {
        vector<double> pos;
        Sp.intpos_to_pos(Sp.P[target].IntPos, pos.da); /* converts the integer coordinate to floating point */

        double r = sqrt(pos.r2());

        double m = All.Mass_StaticHQHalo * pow(r / (r + All.A_StaticHQHalo), 2);

        if(r > 0)
          Sp.P[target].GravAccel += (-All.G * m / (r * r * r)) * pos;

#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
        Sp.P[target].ExtPotential += (-All.G * All.Mass_StaticHQHalo / (r + All.A_StaticHQHalo));
#endif
      }
#endif
    }
}

#endif
