#pragma once
#include <cstddef>
// Exact accepted 80 ms graph geometries, independently verified against the
// selected profile. The 40 ms terminal packet has half this M in every case.
inline std::size_t approved_max_m(std::size_t k,std::size_t n) {
  struct Geometry {std::size_t k,n,m;};
  constexpr Geometry allowed[]={{64,2048,2},{2048,8192,2},{1024,1024,16},
    {1024,3072,16},{512,512,96},{512,1280,96},{256,256,480},
    {128,128,960},{64,64,1920},{32,32,3840}};
  for(auto g:allowed)if(g.k==k && g.n==n)return g.m;
  return 0;
}
