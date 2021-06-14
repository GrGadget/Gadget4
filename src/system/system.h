/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  system.h
 *
 *  \brief declares functions for various low level helper routines
 */

#ifndef SYSTEM_H
#define SYSTEM_H

#include <gsl/gsl_rng.h>
#include <stdio.h>

#include "gadget/mpi_utils.h"  // TAG_HEADER
namespace gadget{
extern gsl_rng *random_generator; /**< the random number generator used */

void myflush(FILE *fstream);

void enable_core_dumps_and_fpu_exceptions(void);

void permutate_chunks_in_list(int ncount, int *list);

void init_rng(int thistask);
double get_random_number(void);
}
#endif
