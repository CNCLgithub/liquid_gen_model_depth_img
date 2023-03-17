# liquid_gen_model_depth_img
## Clone the repo

First, clone this repo:

`git clone git@github.com:CNCLgithub/liquid_gen_model.git`

## Get started with setting up simulation engine

To get started, you need to have SPlisHSPlasH installed inside this repo. You can either go to their website and follow the instructions, as in here: https://splishsplash.readthedocs.io/en/latest/build_from_source.html

Or we provide a way to install, first download this singularity container (https://yale.box.com/shared/static/j40o27bcfgjkzzgecltoh0vj0v411ph5.sif), 

`wget https://yale.box.com/shared/static/j40o27bcfgjkzzgecltoh0vj0v411ph5.sif -O ss.sif`

Then use the following command to install 

`singularity exec ${path_to_downloaded_singularity_container} bash ./setup_ss.sh`. 

where you can have `./ss.sif` instead of `${path_to_downloaded_singularity_container}` if you have been following along.

Remember to install SPlisHSPlasH inside this project directory. i.e., the structure should look like `liquid_gen_model_depth_img/SPlisHSPlasH/` 

## Get started with the model
use `./setup.sh all` to install all the modeling environment. Please note that if packages for Julia cannot be installed in this way, run `./run.sh julia` to open Julia interface and install them manually. 
If you were to install manually, type the following command in the interface: 
```
using Pkg; Pkg.instantiate()
using Pkg; Pkg.add("FileIO")
Pkg.add("GeometryBasics")
Pkg.add("MeshIO")
Pkg.add("PyCall")
Pkg.add("Reexport")
Pkg.add("Formatting")
```

## Run model
To run the model, use `./run.sh julia src/exp_basic.jl ${#}/{Scene_name}`. Please note that {#} corresponding to the number in library. [1-box, 2-boxwithahole, 3-oneobject, 4-obstacle, 5-motor, 6-wall] 
{Scene_names} corresponds to the available scene names. The scene names are combinations of scene names, which are "box" "boxwithahole" "oneobject" "obstacle" "motor" "wall", and viscosities, which are "1016" "104" "1" "4" "16". Connect them using an underscore, such as `box_1016`

As an example, if you were to run box scene at viscosity 4, then run `./run.sh julia src/exp_basic.jl 1/box_4`

A pre-built singularity container can be downloaded here: https://yale.box.com/s/7xjvx27hijjaewezso8l0mbf6hsrq3ha

