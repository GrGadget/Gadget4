/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file dtypes.h
 *
 *  \brief defines some custom data types used by the code
 */

#ifndef DTYPES_H
#define DTYPES_H

#include <cstddef>  // size_t
#include <cstdint>  // int32_t, int64_t, etc
#ifdef EXPLICIT_VECTORIZATION
#include <vectorclass/vectorclass.h>
#endif

#if !defined(POSITIONS_IN_32BIT) && !defined(POSITIONS_IN_64BIT) && !defined(POSITIONS_IN_128BIT)
/* ok, nothing has been chosen as part of the configuration, then use a default value */
#ifndef DOUBLEPRECISION
#define POSITIONS_IN_32BIT
#else
#define POSITIONS_IN_64BIT
#endif
#endif

/* Exactly one of the symbols POSITIONS_IN_32BIT, POSITIONS_IN_64BIT or POSITIONS_IN_128BIT need to be fined, otherwise
 * it is desirable to get a compile time error
 */

#ifdef POSITIONS_IN_32BIT
typedef std::uint32_t MyIntPosType;
typedef std::int32_t MySignedIntPosType;
#define BITS_FOR_POSITIONS 32
#ifdef EXPLICIT_VECTORIZATION
typedef vectorclass::Vec4ui Vec4MyIntPosType;
typedef vectorclass::Vec4i Vec4MySignedIntPosType;
#endif
#endif

#ifdef POSITIONS_IN_64BIT
typedef std::uint64_t MyIntPosType;
typedef std::int64_t MySignedIntPosType;
#define BITS_FOR_POSITIONS 64
#ifdef EXPLICIT_VECTORIZATION
typedef vectorclass::Vec4uq Vec4MyIntPosType;
typedef vectorclass::Vec4q Vec4MySignedIntPosType;
#endif
#endif

#ifdef POSITIONS_IN_128BIT
typedef uint128_t MyIntPosType;
typedef int128_t MySignedIntPosType;
#define BITS_FOR_POSITIONS 128
#ifdef EXPLICIT_VECTORIZATION
#error "EXPLICIT_VECTORIZATION and POSITIONS_IN_128BIT do not work together"
#endif
#endif

#if !defined(IDS_32BIT) && !defined(IDS_48BIT) && !defined(IDS_64BIT)
#define IDS_32BIT
#endif

#ifdef IDS_32BIT
typedef unsigned int MyIDType;
#else
typedef unsigned long long MyIDType;
#endif

#ifdef FOF_ALLOW_HUGE_GROUPLENGTH
typedef long long MyLenType;
#else
typedef int MyLenType;
#endif

#ifdef USE_SINGLEPRECISION_INTERNALLY
typedef float MyReal;
#else
typedef double MyReal;
#endif

#ifndef DOUBLEPRECISION /* default is single-precision */
typedef float MyFloat;
typedef float MyDouble;
typedef float MyNgbTreeFloat;
#define MPI_MYFLOAT MPI_FLOAT
#define MPI_MYDOUBLE MPI_FLOAT
#define H5T_NATIVE_MYFLOAT H5T_NATIVE_FLOAT
#define H5T_NATIVE_MYDOUBLE H5T_NATIVE_FLOAT
#else
#if(DOUBLEPRECISION == 2) /* mixed precision */
typedef float MyFloat;
typedef double MyDouble;
typedef float MyNgbTreeFloat;
#define MPI_MYFLOAT MPI_FLOAT
#define MPI_MYDOUBLE MPI_DOUBLE
#define H5T_NATIVE_MYFLOAT H5T_NATIVE_FLOAT
#define H5T_NATIVE_MYDOUBLE H5T_NATIVE_DOUBLE
#else /* everything double-precision */
typedef double MyFloat;
typedef double MyDouble;
typedef double MyNgbTreeFloat;
#define MPI_MYFLOAT MPI_DOUBLE
#define MPI_MYDOUBLE MPI_DOUBLE
#define H5T_NATIVE_MYFLOAT H5T_NATIVE_DOUBLE
#define H5T_NATIVE_MYDOUBLE H5T_NATIVE_DOUBLE
#endif
#endif

#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
typedef long long integertime;
#define TIMEBINS 60
#define TIMEBASE                                                                                           \
  (((long long)1) << TIMEBINS) /* The simulated timespan is mapped onto the integer interval [0,TIMESPAN], \
                                *  where TIMESPAN needs to be a power of 2. */
#else
typedef int integertime;
#define TIMEBINS 29
#define TIMEBASE (1 << TIMEBINS)
#endif

#ifndef NUMBER_OF_MPI_LISTENERS_PER_NODE
#define NUMBER_OF_MPI_LISTENERS_PER_NODE 1
#endif

#ifndef MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY
#define MAX_NUMBER_OF_RANKS_WITH_SHARED_MEMORY 64
#endif

struct offset_tuple
{
  char n[3];

  offset_tuple() {} /* constructor */

  offset_tuple(const char x) /* constructor  */
  {
    n[0] = x;
    n[1] = x;
    n[2] = x;
  }

  offset_tuple(const char x, const char y, const char z) /* constructor  */
  {
    n[0] = x;
    n[1] = y;
    n[2] = z;
  }
};

struct location
{
  int task;
  int index;
};

inline bool operator==(const location &left, const location &right) { return left.task == right.task && left.index == right.index; }

inline bool operator!=(const location &left, const location &right) { return left.task != right.task || left.index != right.index; }

inline bool operator<(const location &left, const location &right)
{
  if(left.task < right.task)
    return true;
  else if(left.task == right.task)
    {
      if(left.index < right.index)
        return true;
      else
        return false;
    }
  else
    return false;
}

struct halotrees_table
{
  int HaloCount;
  long long FirstHalo;
  long long TreeID;
};

struct parttrees_table
{
  int ParticleCount;
  long long ParticleFirst;
  long long TreeID;
};

struct times_catalogue
{
  double Time;
  double Redshift;
};

enum mysnaptype
{
  NORMAL_SNAPSHOT,
  MOST_BOUND_PARTICLE_SNAPHOT,
  MOST_BOUND_PARTICLE_SNAPHOT_REORDERED
};

enum restart_options
{
  RST_BEGIN,
  RST_RESUME,
  RST_STARTFROMSNAP,
  RST_FOF,
  RST_POWERSPEC,
  RST_CONVERTSNAP,
  RST_CREATEICS,
  RST_CALCDESC,
  RST_MAKETREES,
  RST_IOBANDWIDTH,
  RST_LCREARRANGE,
  RST_SNPREARRANGE
};

struct data_partlist
{
  int Task;  /** The task the item was exported to. */
  int Index; /** The particle index of the item on the sending task. */
};

struct thread_data
{
  int Nexport;
  int NexportNodes;

  double Interactions; /*!< The total cost of the particles/nodes processed by each thread */

  double Ewaldcount; /*!< The total cost for the Ewald correction per thread */
  int FirstExec;     /*!< Keeps track, if a given thread executes the gravity_primary_loop() for the first time */

  std::size_t ExportSpace;
  std::size_t InitialSpace;
  std::size_t ItemSize;

  int *P_CostCount;
  int *TreePoints_CostCount;
  int *Node_CostCount;

  data_partlist *PartList;
  int *Ngblist;
  int *Shmranklist;
  int *Exportflag;
};

#ifdef LONG_X_BITS
#define LONG_X (1 << (LONG_X_BITS))
#define MAX_LONG_X_BITS LONG_X_BITS
#else
#define LONG_X 1
#define MAX_LONG_X_BITS 0
#endif

#ifdef LONG_Y_BITS
#define LONG_Y (1 << (LONG_Y_BITS))
#define MAX_LONG_Y_BITS LONG_Y_BITS
#else
#define LONG_Y 1
#define MAX_LONG_Y_BITS 0
#endif

#ifdef LONG_Z_BITS
#define LONG_Z (1 << (LONG_Z_BITS))
#define MAX_LONG_Z_BITS LONG_Z_BITS
#else
#define LONG_Z 1
#define MAX_LONG_Z_BITS 0
#endif

#define LONG_BITS_MAX(A, B) (((A) > (B)) ? (A) : (B))

#define LEVEL_ALWAYS_OPEN LONG_BITS_MAX(MAX_LONG_X_BITS, LONG_BITS_MAX(MAX_LONG_Y_BITS, MAX_LONG_Z_BITS))

#ifdef GRAVITY_TALLBOX

#if(GRAVITY_TALLBOX == 0)
#define DBX 2
#define DBX_EXTRA 6
#define BOXX (1.0 / LONG_Y)
#define BOXY (1.0 / LONG_Z)
#else
#define DBX 1
#define DBX_EXTRA 0
#endif

#if(GRAVITY_TALLBOX == 1)
#define DBY 2
#define DBY_EXTRA 6
#define BOXX (1.0 / LONG_X)
#define BOXY (1.0 / LONG_Z)
#else
#define DBY 1
#define DBY_EXTRA 0
#endif

#if(GRAVITY_TALLBOX == 2)
#define DBZ 2
#define DBZ_EXTRA 6
#define BOXX (1.0 / LONG_X)
#define BOXY (1.0 / LONG_Y)
#else
#define DBZ 1
#define DBZ_EXTRA 0
#endif

#else

#define DBX 1
#define DBY 1
#define DBZ 1
#define DBX_EXTRA 0
#define DBY_EXTRA 0
#define DBZ_EXTRA 0
#endif

#endif
