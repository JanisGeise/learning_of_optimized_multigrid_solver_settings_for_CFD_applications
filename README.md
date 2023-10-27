# Overview
Master thesis about accelerating multigrid solvers in OpenFoam using DRL.

# Getting started

## General information
*RunFunctions* and *setup-env* taken from [drlfoam](https://github.com/OFDataCommittee/drlfoam), currently maintained by
[@AndreWeiner](https://github.com/AndreWeiner/)

## Test cases
All simulations located in the *test cases* directory are *OpenFOAM* tutorials with minor modifications. The original
cases can be found under:

1. *mixerVesselAMI*:

    `$FOAM_TUTORIALS/multiphase/interFoam/RAS/`


2. *surfaceMountedCube*:

    `$FOAM_TUTORIALS/incmpressible/pimpleFoam/LES/`


3. *weirOverflow*:

    `$FOAM_TUTORIALS/multiphase/interFoam/RAS/`


4. *cylinder2D*:

    `$FOAM_TUTORIALS/incompressible/pimpleFoam/laminar/`

The policy `random_policy.pt` in each simulation directory of the `test_cases` directory is a randomly initialized 
policy created by using the `create_dummy_policy.py` script within [drlfoam](https://github.com/JanisGeise/drlfoam/tree/solver_settings).
The policy `trained_policy.pt` denotes a trained policy, which is already included in the `controlDict` as well.

### Run simulations locally

#### without container

#### with container
-> container can be build by executing   
`sudo singularity build of2206-py1.12.1-cpu.sif docker://andreweiner/of_pytorch:of2206-py1.12.1-cpu`

-> by default, it is assumed that the container is located under `/test_cases/`  
-> directory containing the test cases needs to be located in the `/home/` directory, refer to 
[this](https://github.com/AndreWeiner/ml-cfd-lecture/issues/6) issue for more information  
-> `source setup-env --container`  
-> additionally the MPI version have to match (container uses MPI v. 4.1.2), otherwise this will lead to an error when
executing the flow solver (meshing etc. works)

### Run simulation with SLURM
`module load singularity/latest`  
`./Allwmake --container`

## Installation of drlfoam
drlfoam repository can be found [here](https://github.com/JanisGeise/drlfoam/tree/solver_settings) (branch *solver_settings*)


# Troubleshooting
In case something is not working as expected or if you find any bugs, please feel free to open up a new 
[issue](https://github.com/JanisGeise/learning_of_optimized_multigrid_solver_settings_for_CFD_applications/issues).

# Report