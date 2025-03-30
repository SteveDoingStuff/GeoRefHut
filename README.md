# GeoRefHut--

## Description
GeoRefHut-- is a C++ project for aligning E57 point clouds via GICP, optimizing the resulting pose graph and georeferencing the aligned point clouds.

## Dependencies
- [libE57Format](https://github.com/libE57/libE57Format)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
- [small_gicp](https://github.com/SMIT-Lab/small_gicp)
- [g2o](https://github.com/RainerKuemmerle/g2o)
- GLU (OpenGL Utility Library)
- [tinyfiledialogs](https://sourceforge.net/projects/tinyfiledialogs/)
- [json](https://github.com/nlohmann/json)
- Eigen3

## Setup
Set the following environment variables to point to the respective library directories:
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
```

## Build
### Clone repository:
```bash
git clone https://github.com/your-username/GeoRefHut.git
cd GeoRefHut
```
### Install dependencies:
Just create a folder 'dependencies' in your working directory and git clone and build those libraries there following their respective instructions.

### Build:
```bash
mkdir build
cd build
cmake ..
make
```
### Run:
```bash
./GeoRefHut
```
