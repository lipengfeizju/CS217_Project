# CS217_Project
Point Cloud Classification

## 1. Setup
Conda is recommended to config this repo, more details about Miniconda can be found [here](https://docs.conda.io/en/latest/miniconda.html)
### 1.1 Setup python evironment
```shell
conda create --name gpu_proj python=3.6
conda activate gpu_proj
pip install -r requirements.txt
```
### 1.2 Setup C++ evironment
To build this project in command line, cmake is required, https://cmake.org/download/, after downloading Cmake, don't forget to add its path to `PATH` variable.
```shell
mkdir build && cd build
cmake ..
make -j4
```
For Mac Users, CLion is recommended for debugging CPU part.


## 2. Demo
First we run the C++ module to get ICP result
```shell
./build/registration
```

Then we run the python code to visualize the result

## 3. Configure Workspace
Nsight Eclipse is the default and best IDE to debug CUDA applications as far as I know. It also works perfectly with nvprof to show the porfiling results. The following steps show how to import this project into Nsight Eclipse

### 3.1 Prepare Directory 
```shell
cd .. && mkdir icp_build && cd icp_build
cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=RelWithDebInfo -D USE_VDEC=0 ../icp/
```

Now start up Eclipse and do the following to import the project.
* File→Import...
* General→Existing Projects into Workspace
* For the root directory enter the build root directory, not the source root
Leave other options unchecked and click Finish

### 3.2 Compilation 
First, compile the programs in `icp_build`

```shell
    make
```
Then you have to tell Eclipse what binary to run and what project to compile.

* Go to Run → Run Configurations...
* Double click on C/C++ Application to create a new configuration.
* Browse to the NG executable (build/src/NightshadeNG) for the Application field.
* Browse to select the NG project for the Project field.

### 3.3 Debugging
* Set up a debugging configuration in Eclipse.
* Be sure to manually add the actual source directory in the source tab as a local filesystem path.
* Add source paths to shared library code also so you can step through those.

Then, enjoy the IDE.


