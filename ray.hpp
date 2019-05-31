#ifndef RAYTRACER_IN_ONE_WEEKEND_RAY_HPP
#define RAYTRACER_IN_ONE_WEEKEND_RAY_HPP
#include "vec3.hpp"

class ray
{
  public:
    __device__ vec3 origin() const       { return A; }
    __device__ vec3 direction() const    { return B; }
    __device__ ray() {}
    // __device__ ray(const vec3&amp; a, const vec3&amp; b) { A = a; B = b; }
    __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
    __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

    vec3 A;
    vec3 B;
};

#endif //RAYTRACER_IN_ONE_WEEKEND_RAY_HPP

