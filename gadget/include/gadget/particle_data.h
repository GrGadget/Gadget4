/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file particle_data.h
 *
 *  \brief declares a structure that holds the data stored for a single particle
 */

#ifndef PARTDATA_H
#define PARTDATA_H

#include "gadgetconfig.h"

#include <array>
#include <atomic>
#include <cstring>  // memcpy

#include "gadget/dtypes.h"      // integertime, MyIntPosType
#include "gadget/idstorage.h"   // MyIDStorage
#include "gadget/peano.h"       // peanokey
#include "gadget/symtensors.h"  // vector

/** This structure holds all the information that is
 * stored for each particle of the simulation.
 */
namespace gadget { 
 
struct particle_data
{
  // we do this ugly trick of using memcpy for our own copy constructor and assignment operator
  // because the atomic_flag in particle_data has an implicitly deleted copy operator... so that the implicit functions
  // for this are unavailable. But we know what we are doing here, and surrounding this with an ugly hack
  // is the easiest way at the moment to work around this in our case unnecessary protection

  particle_data() {}

  // declare our own copy constructor
  particle_data(particle_data& other) { memcpy(static_cast<void*>(this), static_cast<void*>(&other), sizeof(particle_data)); }

  // declare our own assignment operator
  particle_data& operator=(particle_data& other)
  {
    memcpy(static_cast<void*>(this), static_cast<void*>(&other), sizeof(particle_data));
    return *this;
  }

  MyIntPosType IntPos[3];    /**< particle position at its current time, stored as an integer type */
  // TODO: compute Vel here
  // MyFloat Vel[3];            
                             /**< particle peculiar velocity, which is equal to
                                  dx/dtau where x is the comoving position and
                                  tau the conformal time. Notice that if we
                                  denote r=ax the physical coordinates and 
                                  dt = a dtau the physical time, it follows that
                                  dr/dt = H r + dx/dtau, therefore Vel = dx/dtau
                                  is not only the absolute velocity in the
                                  coordinates (x,tau) but it is also the
                                  peculiar velocity in the physical coordinates
                                  (r,t) */
  
  MyFloat Momentum[3];       /**< particle momemtum (p) */
  // MyFloat Force[3];          /**< force = dp/dt */
  
  vector<MyFloat> GravAccel; /**< particle acceleration due to gravity */
#if defined(PMGRID) && defined(PERIODIC) && !defined(TREEPM_NOTIMESPLIT)
  MyFloat GravPM[3]; /**< particle acceleration due to long-range PM gravity force */
#endif

  ::std::atomic<integertime> Ti_Current; /**< current time on integer timeline */
  float OldAcc;                        /**< magnitude of old gravitational force. Used in relative opening criterion */
  int GravCost;                        /**< weight factors used for balancing the work-load */

 private:
#ifndef LEAN
  MyDouble Mass; /**< particle mass */
#else
  static MyDouble Mass; /**< particle mass */
#endif
 public:
  MyIDStorage ID;           // 6-byte
  signed char TimeBinGrav;  // 1-byte
#ifndef LEAN
  signed char TimeBinHydro;
#endif
#if defined(MERGERTREE) && defined(SUBFIND)
  compactrank_t PrevRankInSubhalo;  // 1-byte
  MyHaloNrType PrevSubhaloNr;       // 6-byte
  approxlen PrevSizeOfSubhalo;      // 2-byte
#endif

 private:
#ifndef LEAN
  unsigned char Type; /**< flags particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
#else
  static constexpr char Type = 1;
#endif
 public:
#ifndef LEAN
  ::std::atomic_flag access;
#endif

#ifdef REARRANGE_OPTION
  unsigned long long TreeID;
#endif

#if NSOFTCLASSES > 1
 private:
  unsigned char
      SofteningClass : 7; /* we use only 7 bits here so that we can stuff 1 bit for ActiveFlag into it in the Tree_Points structure */
 public:
#endif

#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
  unsigned char InsideOutsideFlag : 1;
#endif

#ifdef FORCETEST
  MyFloat GravAccelDirect[3]; /*!< particle acceleration calculated by direct summation */
  MyFloat PotentialDirect;
  MyFloat DistToID1;
#ifdef PMGRID
  MyFloat GravAccelShortRange[3];
  MyFloat PotentialShortRange;
#ifdef PLACEHIGHRESREGION
  MyFloat GravAccelVeryShortRange[3];
  MyFloat PotentialVeryShortRange;
  MyFloat PotentialHPM;
  MyFloat GravAccelHPM[3];
#endif
#endif
  bool SelectedFlag;
#endif

#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
  MyFloat Potential; /**< gravitational potential */
#if defined(PMGRID)
  MyFloat PM_Potential;
#endif
#ifdef EXTERNALGRAVITY
  MyFloat ExtPotential;
#endif
#endif

#ifdef STARFORMATION
  MyFloat StellarAge;  /**< formation time of star particle */
  MyFloat Metallicity; /**< metallicity of gas or star particle */
#endif

  inline unsigned char getType(void) { return Type; }

  inline unsigned char getTimeBinHydro(void)
  {
#ifndef LEAN
    return TimeBinHydro;
#else
    return 0;
#endif
  }

  inline void setTimeBinHydro(unsigned char bin)
  {
#ifndef LEAN
    TimeBinHydro = bin;
#endif
  }

  inline void setType(unsigned char type)
  {
#ifndef LEAN
    Type = type;
#endif
  }

  inline float getOldAcc(void) { return OldAcc; }

  inline int getGravCost(void) { return GravCost; }

  inline MyDouble getMass(void) { return Mass; }

  inline void setMass(MyDouble mass) { Mass = mass; }

  inline integertime get_Ti_Current(void) { return Ti_Current; }

  inline signed char getTimeBinGrav(void) { return TimeBinGrav; }

  inline unsigned char getSofteningClass(void)
  {
#if NSOFTCLASSES > 1
    return SofteningClass;
#else
    return 0;
#endif
  }

  inline void setSofteningClass(unsigned char softclass)
  {
#if NSOFTCLASSES > 1
    SofteningClass = softclass;
#endif
  }

  // inline double getAscale(void) { return All.Time; }

#if defined(LIGHTCONE_PARTICLES_GROUPS)
  inline void setFlagSaveDistance(void) {}
  inline void clearFlagSaveDistance(void) {}

  inline bool getFlagSaveDistance(void) { return true; }
#endif
};

struct subfind_data
{
  MyHaloNrType GroupNr;
#if defined(MERGERTREE)
  MyHaloNrType SubhaloNr;
  approxlen SizeOfSubhalo;
  compactrank_t RankInSubhalo;
#endif
  char DomainFlag;

  int OriginIndex, OriginTask;
  int TargetIndex, TargetTask;

#ifdef SUBFIND
  int SubRankInGr;

#ifndef SUBFIND_HBT
  struct nearest_ngb_data
  {
    location index[2];
    int count;
  };

  nearest_ngb_data nearest;

  int submark;
  int InvIndex;
#endif

#ifndef LEAN
  int Type;
  MyFloat Utherm;
#endif

#ifdef SUBFIND_STORE_LOCAL_DENSITY
  MyFloat SubfindHsml;     // search radius used for SUBFIND dark matter neighborhood
  MyFloat SubfindDensity;  // total matter density
  MyFloat SubfindVelDisp;  // 3D dark matter velocity dispersion
#endif

  union
  {
    struct
    {
      int originindex, origintask;

      union
      {
        MyFloat DM_Density;
        MyFloat DM_Potential;
      } u;

    } s;

    peanokey Key;
  } u;

  union
  {
    MyFloat DM_Hsml;
    MyFloat DM_BindingEnergy;
  } v;
#else
  /* this are fields defined when we have FOF without SUBFIND */
#ifndef LEAN
  int Type;
#endif
  union
  {
    peanokey Key;
  } u;
#endif
};

// SUBFIND_ORPHAN_TREATMENT
struct idstoredata
{
  int NumPart;
  MyIDType* ID;
};

}
#endif
