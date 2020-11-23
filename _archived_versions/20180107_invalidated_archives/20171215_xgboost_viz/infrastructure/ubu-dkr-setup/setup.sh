#!/usr/bin/env bash

# TODO: SECURE SECRETS

SETUP_DIR=$1
DNS_NAME=$2
DKR_REG_NAME="pcsdevml"

echo "SETUP_DIR: $SETUP_DIR"
echo "DNS_NAME: $DNS_NAME"

WORK_DIR=$(pwd)

apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common cifs-utils

### jmespath command line parsing ###
wget https://github.com/jmespath/jp/releases/download/0.1.3/jp-linux-amd64 -O /usr/local/bin/jp \
        && sudo chmod +x /usr/local/bin/jp

### powershell ###
# see: https://github.com/PowerShell/PowerShell/blob/master/docs/installation/linux.md#ubuntu-1604
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list | tee /etc/apt/sources.list.d/microsoft.list
apt-get update
apt-get install -y powershell

# docker install
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce

# add our container registry
docker login $DKR_REG_NAME.azurecr.io -u pcsdevml -p [pwd]


### create docker volumes for training data (read only?) and result/outputs ###

## mnt azure file storage ##
# https://docs.microsoft.com/en-us/azure/storage/files/storage-how-to-use-files-linux
AZ_STORAGE_NAME="[storage-name]"
AZ_STORAGE_KEY="[key]"
function _mount_az_file_share() {
    SHARE_NAME=$1
    MNT_BASE_DIR=$2

    mkdir -p $MNT_BASE_DIR/$SHARE_NAME

    FSTAB_RECORD="//${AZ_STORAGE_NAME}.file.core.windows.net/${SHARE_NAME} ${MNT_BASE_DIR}/${SHARE_NAME} cifs vers=2.1,username=${AZ_STORAGE_NAME},password=${AZ_STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino"
    echo $FSTAB_RECORD >> /etc/fstab
}
_mount_az_file_share training-data /var/pcsml/azfiles
_mount_az_file_share training-results /var/pcsml/azfiles
# relaod fstab
mount -a

## create docker volumes ##
function _dkr_volume_create() {
    _volume_name=$1
    _mnt_point=$2

    # remote if exists
    if docker volume ls | grep -q ${_volume_name}
    then
        docker volume rm ${_volume_name}
    fi

    # https://github.com/moby/moby/issues/19990#issuecomment-248955005
    docker volume create $_volume_name --opt type=none --opt device=$_mnt_point --opt o=bind
    # print volume info
    docker volume inspect $_volume_name
}
_dkr_volume_create pcsml.training-data /var/pcsml/azfiles/training-data
_dkr_volume_create pcsml.training-results /var/pcsml/azfiles/training-results

### add the remote user to the docker group ###
USER=$(whoami)
usermod -aG docker $USER

# restart dockerd, just in case
systemctl daemon-reload
systemctl enable docker
systemctl restart docker
