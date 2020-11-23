#!/usr/bin/env bash

docker build -t pcsml/sklearn -f Dockerfile-sklearn .
docker build -t pcsml/sklearn-remote -f Dockerfile-sklearn-remote .

_container_name="pcsml-sklearn-remote"
if docker ps | grep -q ${_container_name}
then
    docker container stop ${_container_name}
    docker container rm ${_container_name}
fi

docker container run -d --name ${_container_name} --network=host pcsml/sklearn-remote \
        --mount source=mldev001,target=/var/opt/pcsml/py-ml-spike \
        --mount source-mldata001,target/var/opt/pcsml/ml-data