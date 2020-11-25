# CS217_Project
Point Cloud Classification

## 1. Setup
Conda is recommended to config this repo, more details about Miniconda can be found [here](https://docs.conda.io/en/latest/miniconda.html)
### Setup python evironment
```shell
conda create --name gpu_proj python=3.6
conda activate gpu_proj
pip install -r requirements.txt
```
### Setup C++ evironment
To build this project in command line, cmake is required, https://cmake.org/download/
```shell
mkdir build && cd build
cmake ..
make -j4
```
For Mac Users, CLion is recommended for debugging.


## 2. Demo
First we run the C++ module to get ICP result
```shell
./build/registration
```

Then we run the python code to visualize the result
