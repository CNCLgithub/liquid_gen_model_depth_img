#!/bin/bash

. load_config.sh

usage="Syntax: $(basename "$0") [-h|--help] [COMPONENTS...] -- will set up the project environment,
where:
    -h | --help     Print this help
    COMPONENTS...   Specify component to set up

Valid COMPONENTS:
    all: set up all components (container will be pulled, not built)
    cont_[pull|build]: pull the singularity container or build it
    data: pull data"


if [[ $# -eq 0 ]] || [[ "$@" =~ "--help" ]] || [[ "$@" =~ "-h" ]];then
    echo "$usage"
    exit 0
fi


# container setup
if [[ "$@" =~ "cont_pull" ]] || [[ "$@" =~ "all" ]];then
    echo "Pulling singularity container..."
    wget "" -O "${ENV[cont]}"
elif [[ "$@" =~ "cont_build" ]];then
    echo "Building singularity container..."
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity
else
    echo "Not touching container"
fi


# conda setup
if [[ "${@}" =~ "conda" ]] || [[ "$@" =~ "all" ]];then
    singularity exec ${ENV[cont]} bash -c "yes | conda create -p $PWD/${ENV[env]} python=3.6" && \
    ./run.sh python -m pip install -r requirements.txt
fi


# julia setup
if [[ "${@}" =~ "julia" ]] || [[ "$@" =~ "all" ]];then
    ./run.sh julia -e '"using Pkg; Pkg.instantiate()"'
    ./run.sh julia -e '"using Pkg; Pkg.add('"FileIO"')"'
    ./run.sh julia -e '"using Pkg; Pkg.add('"GeometryBasics"')"'
    ./run.sh julia -e '"using Pkg; Pkg.add('"MeshIO"')"'
    ./run.sh julia -e '"using Pkg; Pkg.add('"PyCall"')"'
    ./run.sh julia -e '"using Pkg; Pkg.add('"Reexport"')"'
    ./run.sh julia -e '"using Pkg; Pkg.add('"Formatting"')"'
fi


# download stimulus set
if [[ "$@" =~ "data" ]] || [[ "$@" =~ "all" ]];then
    echo "Pulling data..."
    mv xxx library
    mkdir "out"
else
    echo "Not pulling any data"
fi
