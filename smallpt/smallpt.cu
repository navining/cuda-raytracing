#include <math.h>    // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdio.h>   //        Remove "-fopenmp" for g++ version < 4.2
#include <stdlib.h>  // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <cuda_runtime.h>
#include <curand.h> 
#include <curand_kernel.h>

#define checkError(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("CUDA error: %s", cudaGetErrorString(err));              \
      printf("Failed to run stmt %s", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)


struct Vec {         // Usage: time ./smallpt 5000 && xv image.ppm
  double x, y, z;    // position, also color (r,g,b)

  __host__ __device__ Vec(double x_ = 0, double y_ = 0, double z_ = 0) {
    x = x_;
    y = y_;
    z = z_;
  }

  __device__ Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }

  __device__ Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }

  __host__ __device__ Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }

  __device__ Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }

  __host__ __device__ Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }

  __device__ double dot(const Vec &b) const {
    return x * b.x + y * b.y + z * b.z;
  }  // cross:

  __host__ __device__ Vec operator%(Vec &b) {
    return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
  }
};

struct Ray {
  Vec o, d;
  __host__ __device__ Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
  double rad;   // radius
  Vec p, e, c;  // position, emission, color
  Refl_t refl;  // reflection type (DIFFuse, SPECular, REFRactive)

  __device__ Sphere(){}

  __device__ Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_)
      : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}

  __device__ double intersect(const Ray &r) const {  // returns distance, 0 if nohit
    Vec op = p - r.o;  // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;

    if (det < 0)
      return 0;
    else
      det = sqrt(det);

    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

Sphere spheres[] = {
    // Scene: radius, position, emission, color, material
    Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(.0, .0, .0), Vec(.75, .25, .25),DIFF),  // Left
    Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(.0, .0, .0), Vec(.25, .25, .75), DIFF), // Rght
    Sphere(1e5, Vec(50, 40.8, 1e5), Vec(.0, .0, .0), Vec(.75, .75, .75), DIFF),  // Back
    Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(.0, .0, .0), Vec(.0, .0, .0), DIFF),        // Frnt
    Sphere(1e5, Vec(50, 1e5, 81.6), Vec(.0, .0, .0), Vec(.75, .75, .75), DIFF),  // Botm
    Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(.0, .0, .0), Vec(.75, .75, .75), DIFF), // Top
    Sphere(16.5, Vec(27, 16.5, 47), Vec(.0, .0, .0), Vec(1, 1, 1) * .999, SPEC),  // Mirr
    Sphere(16.5, Vec(73, 16.5, 78), Vec(.0, .0, .0), Vec(1, 1, 1) * .999, REFR),  // Glas
    Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(.0, .0, .0), DIFF)  // Lite
};

__host__ __device__ inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

__device__ inline bool intersect(const Ray &r, double &t, int &id, Sphere *spheres) {
  double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;

  for (int i = int(n); i--;)
    if ((d = spheres[i].intersect(r)) && d < t) {
      t = d;
      id = i;
    }

  return t < inf;
}

#define RAND(x) curand_uniform_double(x)

__device__ Vec radiance(const Ray &r, int depth, curandState *state, Sphere *spheres) {
  double t;                                // distance to intersection
  int id = 0;                              // id of intersected object
  if (!intersect(r, t, id, spheres)) return Vec();  // if miss, return black
  const Sphere &obj = spheres[id];         // the hit object
  Vec x = r.o + r.d * t, n = (x - obj.p).norm(),
      nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
  double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;  // max refl

  if (++depth > 5)
    if (RAND(state) < p)
      f = f * (1 / p);
    else
      return obj.e;  // R.R.

  if (obj.refl == DIFF) {  // Ideal DIFFUSE reflection
    double r1 = 2 * M_PI * RAND(state), r2 = RAND(state), r2s = sqrt(r2);
    Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(),
        v = w % u;
    Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
    return obj.e + f.mult(radiance(Ray(x, d), depth, state, spheres));
  } else if (obj.refl == SPEC)  // Ideal SPECULAR reflection
    return obj.e +
           f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, state, spheres));

  Ray reflRay(x, r.d - n * 2 * n.dot(r.d));  // Ideal dielectric REFRACTION
  bool into = n.dot(nl) > 0;                 // Ray from outside going in?
  double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl),
         cos2t;

  if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
      0)  // Total internal reflection
    return obj.e + f.mult(radiance(reflRay, depth, state, spheres));

  Vec tdir =
      (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
  double a = nt - nc, b = nt + nc, R0 = a * a / (b * b),
         c = 1 - (into ? -ddn : tdir.dot(n));
  double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re,
         RP = Re / P, TP = Tr / (1 - P);

  return obj.e +
         f.mult(depth > 2
                    ? (RAND(state) < P ?  // Russian roulette
                           radiance(reflRay, depth, state, spheres) * RP
                                       : radiance(Ray(x, tdir), depth, state, spheres) * TP)
                    : radiance(reflRay, depth, state, spheres) * Re +
                          radiance(Ray(x, tdir), depth, state, spheres) * Tr);
}

__global__ void raytracing(Vec *c, int w, int h, int samps, Ray cam, Vec cx, Vec cy, Vec r, Sphere *spheres) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Create random algorithm object
  curandState *state;
  curand_init(123456789, x, 0, state);

  for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++) { // 2x2 subpixel rows
    for (int sx = 0; sx < 2; sx++, r = Vec()) {  // 2x2 subpixel cols
      for (int s = 0; s < samps; s++) {
        double r1 = 2 * RAND(state), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
        double r2 = 2 * RAND(state), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
        Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) + cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
        r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0 , &state, spheres) * (1. / samps);
      }  // Camera rays are pushed ^^^^^ forward to start in interior
      c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
    }
  }
}


#define BLOCK_SIZE 32

int main(int argc, char *argv[]) {
  int w = 1024, h = 768,
      samps = argc == 2 ? atoi(argv[1]) / 4 : 1;              // # samples
  Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());  // cam pos, dir
  Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135, r,
      *c = new Vec[w * h];

  // Allocate device memory
  Vec *c_device;
  cudaMalloc((void **)&c_device, w * h * sizeof(Vec));
  Sphere *s_device;
  cudaMalloc((void **)&s_device, 9 * sizeof(Sphere));

  // Copy host to device
  cudaMemcpy(c_device, c, w * h * sizeof(Vec), cudaMemcpyHostToDevice);
  cudaMemcpy(s_device, &spheres, 9 * sizeof(Sphere), cudaMemcpyHostToDevice);

  // Launch Kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(w/BLOCK_SIZE, h/BLOCK_SIZE); 
  raytracing<<<dimGrid, dimBlock>>>(c_device, w, h, samps, cam, cx, cy, r, s_device);

  cudaDeviceSynchronize();

  // Copy device to host
  cudaMemcpy(c, c_device, w * h * sizeof(Vec), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(c_device);
  cudaFree(s_device);

  FILE *f = fopen("image-cuda.ppm", "w");  // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);

  for (int i = 0; i < w * h; i++)
    fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
