#!/usr/bin/env bash

# docker build -t pcsml/ssh -f ./infrastructure/Dockerfile-ssh ./infrastructure/


function _dkr_container_remove_if_exists() {
    _container_name=$1

    if docker ps -a | grep -q ${_container_name}
    then
        docker container stop ${_container_name}
        docker container rm ${_container_name}
    fi
}

function dkr_build_sklearn() {
    docker build -t pcsml/sklearn -f ./infrastructure/Dockerfile-sklearn ./infrastructure/
}

function dkr_run_ssh() {
    _container_name="pcsml.ssh-ubu1710-001"
    if docker ps -a | grep -q ${_container_name}
    then
        docker container stop ${_container_name}
        docker container rm ${_container_name}
    fi

    #           --mount source=mldev001,target=/var/opt/pcsml/dev \
    #        --mount source=mldata001,target=/var/opt/pcsml/data \
    docker container run -d --name ${_container_name} \
            --network=host \
            pcsml/ssh
}

function dkr_run_dev() {
    _container_name="pcsml.sklearn-dev"

    _dkr_container_remove_if_exists ${_container_name}

    #           --mount source=mldev001,target=/var/opt/pcsml/dev \
    #        --mount source=mldata001,target=/var/opt/pcsml/data \
    docker container run -d --init --name ${_container_name} \
            --network=host \
            --mount source=pcs-dmf-001-dev,target=/var/opt/devel \
            --mount source=mldata001,target=/var/opt/pcsml/data \
            --mount source=ml_out_001,target=/var/opt/pcsml/out
            pcsml/sklearn-dev
}

function dkr_mbp_run_gis_pps_crossval() {
    docker build -t pcsml/gis_pps_cross_val -f ./Dockerfile-scratch .

    docker container run -id \
            --mount source=mldata001,target=/var/opt/pcsml/data \
            --mount source=ml_out_001,target=/var/opt/pcsml/out \
            pcsml/gis_pps_cross_val
}

#dkr_run_dev
dkr_mbp_run_gis_pps_crossval