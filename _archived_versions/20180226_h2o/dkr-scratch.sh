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

function _dkr_volume_remove_if_exists() {
    _volume_name=$1

    if docker volume ls | grep -q ${_volume_name}
    then
        docker volume rm ${_volume_name}
    fi
}

function dkr_build_base() {
    docker build -t pcsml/base -f ./infrastructure/Dockerfile-pcsml ./infrastructure/
}

function dkr_build_dev() {
    docker build -t pcsml/dev -f ./infrastructure/Dockerfile-pcsml-dev ./infrastructure/
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
    _container_name="pcsml.dev"
#    _dev_dir="/mnt/net/pcs-dmf-001/dev"

    _dkr_container_remove_if_exists ${_container_name}

    #           --mount source=mldev001,target=/var/opt/pcsml/dev \
    #        --mount source=mldata001,target=/var/opt/pcsml/data \
    #            --mount type=bind,src=${_dev_dir},dst=/var/opt/devel \
#            --mount source=mldata001,target=/var/opt/pcsml/data \
#            --mount source=ml_out_001,target=/var/opt/pcsml/out \

    docker container run -d --init --name ${_container_name} \
            --mount type=bind,source=c:/,target=/mnt/c \
            --mount type=bind,source=c:/dev/pcs/ml/py-scikit-spike,target=/var/opt/pcsml/devel \
            -p 42202:42202 \
            pcsml/dev
}

function dkr_mbp_run_main() {
#    _dev_dir="/mnt/net/pcs-dmf-001/dev"
    _result_out_dir="/var/opt/pcsml/out"

    docker build -t pcsml/dev-main -f ./Dockerfile-scratch .

    #--mount source=ml_out_001,target=/var/opt/pcsml/out \
    docker container run -id \
            --mount source=mldata001,target=/var/opt/pcsml/data \
            --mount type=bind,src=${_result_out_dir},target=/var/opt/pcsml/out \
            pcsml/dev-main
}

#dkr_run_dev
#dkr_mbp_run_gis_pps_crossval
#dkr_build_dev

if [ ! -z "$1" ]; then
    echo "running: $1"
    $1
fi