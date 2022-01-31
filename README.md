<div align="center">
  <h1 align="center">Parallel and Distributed Systems Assignment 3</h1>
  <h3 align="center">Aristotle University of Thessaloniki</h3>
  <h4 align="center">School of Electrical & Computer Engineering</h4>
  <p align="center">
    Contributors: Kyriafinis Vasilis, Koro Erika
    <br/>
    Winter Semester 2021 - 2022
    <br />
    <br />
    <br />
    <br />
  </p>
</div>

- [1. About this project](#1-about-this-project)
- [2. Getting started](#2-getting-started)
- [3. Dependencies](#3-dependencies)
  - [3.1. Access to a machine with NVIDIA GPU](#31-access-to-a-machine-with-nvidia-gpu)
  - [3.2. Make](#32-make)
- [4. Project directory layout](#4-project-directory-layout)
  - [4.1. Top-level](#41-top-level)
  - [4.2. project](#42-project)
- [5. Compile and run](#5-compile-and-run)
  - [5.1. Local computer](#51-local-computer)
  - [5.2. Aristotle University HPC](#52-aristotle-university-hpc)

## 1. About this project
Implement in CUDA the evolution of an Ising model in two dimensions for a given number of steps k.

The Ising model, named after the physicist Ernst Ising, is a mathematical model of ferromagnetism in statistical mechanics. The model consists of discrete magnetic dipole moments of atomic “spins” that can be in one of two states (+1 or −1). The spins are arranged in a square 2D lattice with periodic boundary conditions, allowing each spin to interact with its four immediate neighbors. The dipole moments update in discrete time steps according to the majority of the spins among the four neighbors of each lattice point. The edge lattice points wrap around to the other side (known as toroidal or periodic boundary conditions), i.e. if side length is n, and grid coordinates are 0:n−1, then the previous to node 0 is node n−1 and vice versa.

The magnetic moment gets the value of the majority of the spins of its neighbors and itself:

sign(G[i-1,j] + G[i,j-1] + G[i,j] + G[i+1,j] + G[i,j+1]).





## 2. Getting started

To setup this repository on your local machine run the following commands on the terminal:

```console
git clone https://github.com/Billkyriaf/pds_assignment_3.git
```

Or alternatively [*download*](https://github.com/Billkyriaf/pds_assignment_3/archive/refs/heads/main.zip) and extract the zip file of the repository
<br/>

## 3. Dependencies
### 3.1. Access to a machine with NVIDIA GPU

This project is written in CUDA C. In order to compile and run the code a machine with nvida toolkit is requiderd (nvcc, nvprof etc). Instructions on how to install and setup such an enviroment are widely available on the internet. In order to run CUDA the GPU must also support it. 

### 3.2. Make

GNU Make is a tool which controls the generation of executables and other non-source files of a program from the program's source files. This is not strictly required but if you don't have it you must compile and run the project on your own.

<br/>

## 4. Project directory layout

### 4.1. Top-level
```
.
├─ project               # The project files and build scripts
├─ scripts               # Scripts related to the HPC
└─ README.md
```
### 4.2. project
```
.
├── ...
├── project
|   ├── src
|   |   ├── sequential.c           # C sequential implementation
|   |   ├── GPU_single_moment.cu   # First CUDA implementation. One moment per thread
|   |   ├── GPU_block_moments.cu   # Second CUDA implementation. Multiple moments per thread
|   |   └── GPU_block_moments.cu   # Third CUDA implementation. Multiple moments per thread using shared memory 
|   |                                per GPU block
|   |
|   └── Makefile             # Makefile 
|
└─ ...
```
<br/>
<br/>

## 5. Compile and run

### 5.1. Local computer

1. `cd` to the project directory and run one of the following `Makefile` targets:

    - `make all`: Runs all the targets this can take some time depending on the hardware.
    - `make run_sequential`: Compiles and runs a job for the `sequential.c` file.
    - `make run_single_moment`: Compiles and runs a job for the `GPU_single_moment.cu` file.
    - `make run_block_moments`: Compiles and runs a job for the `GPU_block_moments.cu` file.
    - `make run_shared_memory`: Compiles and runs a job for the `GPU_shared_mem.cu` file.

### 5.2. Aristotle University HPC

To run the project on the HPC `git clone` the project while connected to the HPC with ssh.

1. Load the modules required. Run `module load gcc/10.2.0 cuda/11.1.0`. The modules must be loaded only once.
2. `cd` to the project directory and run one of the following `Makefile` targets:

    - `make run_sequential_hpc`: Compiles and submits<sup>1</sup> a job for the `sequential.c` file.
    - `make run_single_moment_hpc`: Compiles and submits<sup>1</sup> a job for the `GPU_single_moment.cu` file.
    - `make run_block_moments_hpc`: Compiles and submits<sup>1</sup> a job for the `GPU_block_moments.cu` file.
    - `make run_shared_memory_hpc`: Compiles and submits<sup>1</sup> a job for the `GPU_shared_mem.cu` file.

----
Notes: 

<sup>1</sup> Submitting a job does not mean that the code will run instantly. If there are available resources the `slurm-*.out` file will be created. Otherwise the `make` command will end with error because the file will not be created until the code starts running. You can check the job queue with `squeue | grep gpu` command.

<sup>2</sup> The default time that the `make` target will `tail` the output file is 10s. If the execution takes more than that you can run `tail -fn +1 *.out` and follow the output as long as it's needed.

<sup>3</sup> All of the above targets are run with `nvprof` so inforamtion about the execution times is provided.