#include <iostream>
// #include <values.h>
#include "vec3.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hitable_list.hpp"

__device__ vec3 color_ray(const ray& r, hitable **world) {
  const vec3 white = vec3(1.0, 1.0, 1.0);
  const vec3 blue = vec3(0.5, 0.7, 1.0);
  const vec3 red = vec3(1, 0, 0);

  hit_record record;
  if ( (*world)->hit(r, 0.0, MAXFLOAT, record) ) {
    return 0.5 * vec3(record.normal.x() + 1, record.normal.y() + 1, record.normal.z() + 1);
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * white + t * blue;
  }
}
__global__ void generate(int *A, int nx, int ny,
                         int size, vec3 lower_left_corner,
                         vec3 horizontal, vec3 vertical,
                         vec3 origin, hitable **world){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  float u = float(i) / float(nx);
  float v = float(j) / float(ny);
  ray r(origin, lower_left_corner + u*horizontal + v*vertical);
  vec3 color = color_ray(r, world);
  int ir = int(255.99*color.r());
  int ig = int(255.99*color.g());
  int ib = int(255.99*color.b());
  A[(i*ny + j)*3] = ir;
  A[(i*ny + j)*3 + 1] = ig;
  A[(i*ny + j)*3 + 2] = ib;
}

__global__ void create_world(hitable **d_list, hitable **d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
      *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
      *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
      *d_world    = new hitable_list(d_list,2);
  }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
  delete *(d_list);
  delete *(d_list+1);
  delete *d_world;
}

int main() {
  int nx = 200;
  int ny = 100;
  int size = nx*ny*3*sizeof(int);

  dim3 dimGrid(ceil(nx/(float)16), ceil(ny/(float)16));
  dim3 dimBlock(16, 16);

  int *cpuA;
  int *gpuA;
  cpuA = (int *)malloc(size);
  cudaMalloc((void **)&gpuA,size);

  vec3 lower_left_corner(-2.0, -1.0, -1.0);
  vec3 horizontal(4.0, 0.0, 0.0);
  vec3 vertical(0.0, 2.0, 0.0);
  vec3 origin(0.0, 0.0, 0.0);

  // hitable *list[2];
  hitable** d_list;
  cudaMalloc((void **)&d_list, 2*sizeof(hitable *));

  // hitable *world = new hitable_list(list, 2);
  hitable** d_world;
  cudaMalloc((void **)&d_world, sizeof(hitable *));

  create_world<<<1,1>>>(d_list, d_world);
  
  generate<<<dimGrid, dimBlock>>>(gpuA, nx, ny, size, 
                                  lower_left_corner,
                                  horizontal, vertical, 
                                  origin, d_world);

  cudaMemcpy(cpuA, gpuA, size, cudaMemcpyDeviceToHost);

  cudaFree(gpuA);
  cudaFree(d_list);
  cudaFree(d_world);

  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--){
      for(int i = 0; i < nx; i++){
          std::cout << cpuA[(i*ny + j)*3] << " " << cpuA[(i*ny + j)*3 + 1] << " " << cpuA[(i*ny + j)*3 + 2] << "\n";
      }
  }
  delete[] cpuA;
}
