#!/usr/bin/env bash

VM_ID=$1
VM_ENV="dev"
VM_NAME="pcs-$VM_ENV-ml-dkrh-$VM_ID"

# check if we need to login
_login_err=$(az account show 2>&1 | grep "az login")
if [ $_login_err ]; then
    az login
fi

# set the active subscription to use (could be sent in as param)
az account set --subscription MSDN\ -\ VS\ Prof\ -\ DMF\ -\ 2017-03
az vm create --resource-group "pcs-dev-ml-v02" --name $VM_NAME --image UbuntuLTS --size Standard_B1ms \
    --admin-username [devusername] \
    --nsg pcs-dev-ml-dkrh-nsg \
    --public-ip-address-dns-name "$VM_NAME"