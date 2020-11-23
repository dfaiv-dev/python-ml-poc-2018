#!/usr/bin/env bash

# invoke with something like (make sure you have run `az account set --subscription [name|id]`):
#. ./run_docker_image.sh  -d ./ -s debug002 --no-shutdown -- python data/scripts/df_gis_pps_prepare.py xgb_train.py --run-id debug002

# see: https://stackoverflow.com/a/29754866/79113
OPTIONS=d:c:bg:n:r:s:z:
LONGOPTIONS=dev-dir:,container-name:,build-image,build-base,vm-group:,vm-name:,session-name:,vm-size,no-shutdown
PARSED_OPTS=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")

if [[ -z $1 ]]; then
    echo "see script for usage"
    return
fi

eval set -- "$PARSED_OPTS"
while true; do
    case "$1" in
        -d|--dev-dir)
            dev_dir=$2
            shift 2
            ;;
        -c|--container-name)
            container_name=$2
            shift 2
            ;;
        -b|--build-image)
            build_image=1
            shift
            ;;
        --build-base)
            build_base=1
            shift
            ;;
        -g|--vm-group)
            vm_group=$2
            shift 2
            ;;
        -n|--vm-name)
            vm_name=$2
            shift 2
            ;;
        -z|--vm-size)
            vm_size=$2
            shift 2
            ;;
        -s|--session-name)
            session_name=$2
            shift 2
            ;;
        --no-shutdown)
            no_shutdown="--no-shutdown"
            shift
            ;;
        --)
            shift
            cmd="$@"
            break
            ;;
        *)
            echo "invalid usage"
            ;;
    esac
done

INFRASTRUCTURE_SCRIPTS_DIR=$dev_dir/infrastructure
DOCKER_FILE=$dev_dir/Dockerfile
DOCKER_WORK_DIR=$dev_dir
IMAGE_NAME=pcsml/py-scikit-spike
SESSION_NAME=${session_name-_scratch}
VM_SIZE=${vm_size}
NO_SHUTDOWN=${no_shutdown-"-"}

VM_NAME=${vm_name-"pcs-dev-ml-dkrh-003a"}
VM_REGION=eastus
VM_DNS_NAME="$VM_NAME.$VM_REGION.cloudapp.azure.com"
VM_RESOURCE_GRP=${vm_group-"pcs-dev-ml"}

CTR_REG=pcsdevml.azurecr.io

#sudo apt-get update && sudo apt-get install --only-upgrade -y azure-cli

#docker login $CTR_REG -u pcsdevml -p [pwd]

docker build -t pcsml/base -f ./infrastructure/Dockerfile-pcsml ./infrastructure/
docker tag pcsml/base $CTR_REG/pcsml/base
docker push $CTR_REG/pcsml/base

docker build -t $IMAGE_NAME -f $DOCKER_FILE $DOCKER_WORK_DIR
docker tag $IMAGE_NAME $CTR_REG/$IMAGE_NAME
docker push $CTR_REG/$IMAGE_NAME

### az cli helpers
function _get_vm_state() {
    az vm show \
        --resource-group $VM_RESOURCE_GRP --name $VM_NAME --show-details --query "powerState"
}

function _get_vm_size() {
    az vm show \
        --resource-group $VM_RESOURCE_GRP --name $VM_NAME --query hardwareProfile.vmSize -o tsv
}

function _get_vm_ip() {
    az vm show --show-details \
        --resource-group $VM_RESOURCE_GRP --name $VM_NAME --query publicIps -o tsv
}

### update vm size if needed
_vm_size=$(_get_vm_size)
if [[ ! -z $VM_SIZE ]]; then
    echo "attempting to resize vm to: $VM_SIZE"
    if [[ "$_vm_size" != "$VM_SIZE" ]]; then
        echo "resizing vm from: $_vm_size, to: $VM_SIZE"
        az vm resize \
            --resource-group $VM_RESOURCE_GRP --name $VM_NAME \
            --size $VM_SIZE
    else
        echo "vm is already sized: $_vm_size"
    fi
else
    echo "vm size arg not set, will not change.  Running with size: $_vm_size"
fi

### vm startup
# check if we need to login
echo "checking login status (then power state)"
_login_err=$(az account show 2>&1 | grep "az login")
if [ $_login_err ]; then
    az login
fi

echo "checking power state"
_power_state=$(_get_vm_state)
_is_deallocated=false
if [[ ! -z $(echo "${_power_state,,}" | grep "deallocated") ]]; then
    _is_deallocated=true
fi
_is_running=false
if [[ ! -z $(echo "${_power_state,,}" | grep "running") ]]; then
    _is_running=true
fi
echo "vm powerstate: $_power_state"
echo "vm deallocated: $_is_deallocated"
echo "vm running: $_is_running"

if [[ ! $_is_deallocated = true ]]; then
# this is annoying, turn it off for now.  hopefully the dev knows what she or he is doing.
#    echo -n "vm is active, are you sure you want to deploy? (y/Y, or else any other key, then [ENTER]): "
#    read deploy_override

    deploy_override='y'

    if [  ${deploy_override,,} != 'y' ]; then
        echo "vm active and override not selected.  exiting."
        exit -1
    else
        echo "will execute on running vm."
    fi
fi
if [[ ! $_is_running = true ]]; then
    az vm start --resource-group $VM_RESOURCE_GRP --name $VM_NAME
    echo "vm state: $(_get_vm_state)"
    echo "sleeping to hopefully let vm finish boot..."
    sleep 20
fi

_vm_ip=$(_get_vm_ip)
echo "vm ip: $_vm_ip"

### deploy scripts and run docker container
# do/while loop emulation: https://stackoverflow.com/a/16491478/79113
while
    echo "copying remote exec scripts"
    scp -o StrictHostKeyChecking=no -r $INFRASTRUCTURE_SCRIPTS_DIR/remote-exec  [devusername]@$_vm_ip:~/
    (( $? > 0 ))
do
    continue
done

_docker_run_opts=\
"--mount source=pcsml.training-data,target=/var/opt/pcsml/training-data \
--mount source=pcsml.remote-exec-out,target=/var/opt/pcsml/remote-exec-out"


echo "will try to run docker with command: $cmd"
ssh -oStrictHostKeyChecking=no -t [devusername]@$_vm_ip "bash ~/remote-exec/run-dkr-image.sh \
    $CTR_REG/$IMAGE_NAME "\""$_docker_run_opts"\"" "\""$cmd"\"" \
    ~/remote-exec "\""$VM_RESOURCE_GRP"\"" "\""$VM_NAME"\"" \
    '$SESSION_NAME' '$NO_SHUTDOWN'"

#ssh -t [devusername]@$VM_DNS_NAME \
#"bash ~/remote-exec/run-and-shutdown.sh -c 'echo hello!; sleep 5; echo \$(date)' -g pcs-dev-ml-v02 -n pcs-dev-ml-dkrh-001a"

#ssh -t [devusername]@$VM_DNS_NAME \
#"bash ~/remote-exec/run-tmux.sh debug-session \". ~/remote-exec/run-and-shutdown.sh -c 'echo hello!; sleep 5; echo \$(date)' -g pcs-dev-ml-v0