/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  pm.h
 *
 *  \brief definition of a class to bundle the PM-force calculation algorithms
 */

#ifndef PM_H
#define PM_H

#if defined(PMGRID) || defined(NGENIC)

#include <fftw3.h>

#include "../pm/pm_nonperiodic.h"
#include "../pm/pm_periodic.h"
#include "gadget/setcomm.h"

#ifdef PERIODIC
class pm : public pm_periodic
{
 public:
  pm(MPI_Comm comm) : pm_periodic(comm) {}
};
#else
class pm : public pm_nonperiodic
{
 public:
  pm(MPI_Comm comm) : pm_nonperiodic(comm) {}
};
#endif

#endif

#endif
