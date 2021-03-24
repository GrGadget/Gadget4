#pragma once

#include <array>
#include <cstdlib>

namespace gadget::pm
{
/* General API for Gadget's PM */
class particle_handler
{
 public:
  virtual std::size_t size() const                                    = 0;  // number of particles
  virtual std::array<long long int, 3> get_position(int i) const      = 0;
  virtual double get_mass(int i) const                                = 0;
  virtual void set_acceleration(int i, std::array<double, 3> A) const = 0;
  virtual ~particle_handler() {}
};

}  // namespace gadget::pm
