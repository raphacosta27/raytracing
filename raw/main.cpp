#include <iostream>
// #include <values.h>
#include "vec3.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hitable_list.hpp"
#include <x86intrin.h> //Extensoes SSE
#include <bits/stdc++.h> //Bibliotecas STD
using namespace std;
using namespace std::chrono;

vec3 color_ray(const ray& r, hitable *world) {
  const vec3 white = vec3(1.0, 1.0, 1.0);
  const vec3 blue = vec3(0.5, 0.7, 1.0);
  const vec3 red = vec3(1, 0, 0);

  hit_record record;
  if (world->hit(r, 0.0, MAXFLOAT, record)) {
    return 0.5 * vec3(record.normal.x() + 1, record.normal.y() + 1, record.normal.z() + 1);
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * white + t * blue;
  }
}

int main() {
  auto now = high_resolution_clock::now();
  int nx = 2400;
  int ny = 1600;
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  vec3 lower_left_corner(-2.0, -1.0, -1.0);
  vec3 horizontal(4.0, 0.0, 0.0);
  vec3 vertical(0.0, 2.0, 0.0);
  vec3 origin(0.0, 0.0, 0.0);

  hitable *list[2];
  list[0] = new sphere(vec3(0, 0, -1), 0.5);
  list[1] = new sphere(vec3(0, -100.5, -1), 100);
  hitable *world = new hitable_list(list, 2);
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      float u = float(i) / float(nx);
      float v = float(j) / float(ny);
      ray r(origin, lower_left_corner + u*horizontal + v*vertical);
      vec3 color = color_ray(r, world);
      // I believe 255.99 is used instead of 255 since we're using the `int` constructor on floats, so we'll truncate the results
      //     0 <= r <= 1
      // =>  0 <= 255.99 * r <= 255.99
      // =>  0 <= int(255.99 * r) <= 255
      // It's a nasty way of rounding really, biasing towards rounding down.
      int ir = int(255.99*color.r());
      int ig = int(255.99*color.g());
      int ib = int(255.99*color.b());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }
  auto end_time = duration_cast<duration<double>>(high_resolution_clock::now() - now).count();
  std::cout << "Time: " << end_time << "\n";
  return 0;
}