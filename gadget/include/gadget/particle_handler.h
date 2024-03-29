#pragma once

#include <array>
#include <cstdlib>
#include <gadget/idstorage.h>

namespace gadget
{
/* General API for Gadget's PM */
class particle_handler
{
 public:
  virtual std::size_t size() const                                    = 0;  // number of particles
  virtual std::array<MyIntPosType, 3> get_IntPosition(int i) const   = 0;
  virtual std::array<MyFloat, 3> get_position(int i) const   = 0;
  virtual std::array<MyFloat, 3> get_momentum(int i) const   = 0;
  virtual MyIDType get_id(int i) const   = 0;
  virtual double get_mass(int i) const                                = 0;
  virtual void set_acceleration(int i, std::array<MyFloat, 3> A) const = 0;
  // TODO: velocity here
  // virtual void set_velocity(int i, std::array<MyFloat, 3> V) const = 0;
  
  #ifdef GEVOLUTION_GR
  virtual void set_metric(int i, MyFloat Phi, std::array<MyFloat,3> B) const = 0;
  #endif
  
  virtual ~particle_handler() {}
};

}  // namespace gadget
