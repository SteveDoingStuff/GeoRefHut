# GeoRefHut++
![GUI of GeoRefHut](https://github.com/SteveDoingStuff/GeoRefHut--/blob/main/img/GUI.png)

## Description
GeoRefHut++ is a C++ project for aligning E57 point clouds via GICP, optimizing the resulting pose graph and georeferencing the aligned point clouds. This project has only been tested on Ubuntu 24.04.2 LTS.

## Dependencies
- [libE57Format](https://github.com/asmaloney/libE57Format)
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
- [small_gicp](https://github.com/koide3/small_gicp)
- [g2o](https://github.com/RainerKuemmerle/g2o)
- GLU (OpenGL Utility Library)
- [tinyfiledialogs](https://github.com/native-toolkit/libtinyfiledialogs)
- [json](https://github.com/nlohmann/json)
- Eigen3


## Build
### Clone repository:
```bash
git clone https://github.com/SteveDoingStuff/GeoRefHut.git
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
before running the programm, you might need to set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
```
run program by:
```bash
./GeoRefHut
```
