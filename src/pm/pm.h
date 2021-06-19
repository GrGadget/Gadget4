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

#include <fftw3.h>
#include <array>

#include "../pm/pm_nonperiodic.h"
#include "gadget/pm_periodic.h"
#include "gadget/setcomm.h"
#include "gadget/addons/gevolution-pm.hpp"

namespace gadget{

#ifndef GEVOLUTION_PM
    #ifdef PERIODIC
        class pm : public pm_periodic
        {
         public:
          pm(MPI_Comm comm, std::array<int, 3> ngrid) : pm_periodic(comm, ngrid) {}
        };
    #else
        class pm : public pm_nonperiodic
        {
         public:
          pm(MPI_Comm comm, int ngrid) : pm_nonperiodic(comm, ngrid) {}
        };
    #endif
#else
    class pm : public ::gadget::gevolution_api::newtonian_pm
    {
        using base_t = ::gadget::gevolution_api::newtonian_pm;
     public:
      pm(MPI_Comm comm, int ngrid) : base_t(comm, ngrid) {}
    };
#endif
}
#endif
