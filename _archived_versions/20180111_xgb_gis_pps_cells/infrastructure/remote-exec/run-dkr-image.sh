#!/usr/bin/env bash

IMAGE_NAME=$1
RUN_OPTS=$2
RUN_CMD=$3
SCRIPT_LIB_DIR=$4
VM_GROUP=$5
VM_NAME=$6
SESSION_NAME=$7
NO_SHUTDOWN=$8

function _dkr_container_remove_if_exists() {
    _container_name=$1

    if docker ps -a | grep -q ${_container_name}
    then
        echo "docker container exists ($_container_name), removing"
        docker container stop ${_container_name}
        docker container rm ${_container_name}
    fi
}

if [[ -z $IMAGE_NAME ]]; then
    echo "must supply image name as first arg.  opt args: 2) run opts."
    exit -1
fi

echo "running dkr image: $IMAGE_NAME"
echo "run opts: $RUN_OPTS"

REMOTE_EXEC_OUT_BASEDIR='/var/pcsml/azfiles/remote-exec-out'
if [ ! -d "$REMOTE_EXEC_OUT_BASEDIR" ]; then
    echo "$REMOTE_EXEC_OUT_BASEDIR is not mounted"
    exit -1
fi

docker pull $IMAGE_NAME

_timestamp=$(date +'%Y%m%d_%H%M')
_out_dir="$REMOTE_EXEC_OUT_BASEDIR/$SESSION_NAME"
mkdir -p  $_out_dir

### run the docker container
CONTAINER_NAME="${SESSION_NAME}"
_dkr_container_remove_if_exists $CONTAINER_NAME
echo "run cmd: $RUN_CMD"
echo "run -d --name $CONTAINER_NAME $RUN_OPTS $IMAGE_NAME $RUN_CMD"
docker run -d --name $CONTAINER_NAME $RUN_OPTS $IMAGE_NAME $RUN_CMD

if [[ "$NO_SHUTDOWN" != "--no-shutdown" ]]; then
    echo "vm will shutdown after container runs, (got: $NO_SHUTDOWN)"
    echo "checking az login"
    _login_err=$(az account show 2>&1 | grep "az login")
    if [ $_login_err ]; then
        az login
    fi

    # ahoffmanSubscription                             AzureCloud   49c7d034-2a69-4cac-b9c7-fae4e8092034
    # Visual Studio Professional - David Faivre        AzureCloud   2b1f92db-0edf-4317-94ab-53eca82df9dc
    # MSDN - VS Prof - DMF - 2017-03                   AzureCloud   d1d829db-9a6e-4dec-a028-195c49c8871a
    # greg osb: 7090af32-6d37-4671-abdf-9a1aaf2d3b66
    _subId='7090af32-6d37-4671-abdf-9a1aaf2d3b66'
    az account set --subscription $_subId
    if [[ ! $? -eq 0 ]]; then
        echo "trying logout and login to reset credentials"
        az logout
        az login
        az account set --subscription $_subId
    fi

    _shutdown_cmd="$SCRIPT_LIB_DIR/dkr-monitor-and-shutdown.sh -c $CONTAINER_NAME -g $VM_GROUP -n $VM_NAME | tee $_out_dir/docker-shutdown-logs_${_timestamp}.txt"
    tmux new-session -d -s "${SESSION_NAME}_shutdown_monitor" "$_shutdown_cmd"
else
    echo "********"
    echo "no shutdown set, got: $NO_SHUTDOWN"
    echo "will run container and leave VM running"
    echo "********"
fi

### write docker logs, using tmux session to keep alive
_docker_log_cmd="docker logs -t -f $CONTAINER_NAME | tee $_out_dir/docker-logs_${_timestamp}.txt"
tmux new-session -d -s "${SESSION_NAME}_logs" "$_docker_log_cmd"
tmux ls

docker logs -t -f $CONTAINER_NAME