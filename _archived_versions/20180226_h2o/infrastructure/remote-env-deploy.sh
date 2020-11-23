#!/usr/bin/env bash

docker build -t pcsml/sklearn -f Dockerfile-sklearn .
docker build -t pcsml/sklearn-remote -f Dockerfile-sklearn-remote .
docker build -t pcsml/py-scikit-spike -f Dockerfile-remote-dev .

_container_name="pcsml-py-scikit-spike"
if docker ps -a | grep -q ${_container_name}
then
    docker container stop ${_container_name}
    docker container rm ${_container_name}
fi




docker container run -d --name ${_container_name} \
        --network=host \
        --mount source=mldev001,target=/var/opt/pcsml/dev \
        --mount source=mldata001,target=/var/opt/pcsml/data \
        pcsml/py-scikit-spike