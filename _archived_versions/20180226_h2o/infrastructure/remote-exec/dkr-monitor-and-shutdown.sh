#!/usr/bin/env bash

###
# monitors a container for when it is done running
# saves logs to file
# optional shutdown of vm
###

usage="
    -c | --container-name   the name of the container to monitor
    -g | --vm-group         the azure group the vm is in
    -n | --vm-name          the vm name
    -h | --help             show this
"

# see: https://stackoverflow.com/a/29754866/79113
OPTIONS=c:g:n:h
LONGOPTIONS=container-name:,vm-group:,vm-name:,help
PARSED_OPTS=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")

if [[ -z $1 ]]; then
    echo "$usage"
    return
fi

if [[ $? -ne 0 ]]; then
    # e.g. $? == 1
    #  then getopt has complained about wrong arguments to stdout
    echo "$usage"
    return
fi

eval set -- "$PARSED_OPTS"
while true; do
    case "$1" in
        -c|--container-name)
            container_name=$2
            shift 2
            ;;
        -g|--vm-group)
            vm_group=$2
            shift 2
            ;;
        -n|--vm-name)
            vm_name=$2
            shift 2
            ;;
        -h|--help)
            echo "$usage"
            shift 1
            ;;
        --)
            break
            ;;
        *)
            echo "invalid usage"
            echo "$usage"
            ;;
    esac
done

function _vm_show() {
    az vm show --resource-group $vm_group --name $vm_name \
            --query "{name:name,grp:resourceGroup,profile:hardwareProfile}"
}

function _vm_deallocate() {
    az vm deallocate --resource-group $vm_group --name $vm_name --no-wait
}

# login to azure using pcs-dev-ml-automation service principal user
# https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest
# https://docs.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli?view=azure-cli-latest
# turns out I can't add service principals to other dev's az account :(  DMF 2017-12-30
# login from the script that calls this one...
#echo "az login:"
#az login --service-principal \
#    --username [uname] --password "[pwd]!" \
#    --tenant [tenant-uid]

echo "running on vm:"
az vm show --resource-group $vm_group --name $vm_name \
            --query "{name:name,grp:resourceGroup,profile:hardwareProfile}"

echo
echo "container logs:"
docker attach --no-stdin $container_name

echo
echo "deallocating vm will start in 15 sec, will disconnect any remote ssh..."
sleep 15
_vm_deallocate

echo
echo "press [Enter] to exit, or just wait for the machine to shutdown..."
read