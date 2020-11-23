#!/usr/bin/env bash

# TODO: SECURE SECRETS

SETUP_DIR=$1
DNS_NAME=$2
DKR_REG_NAME="pcsdevml"

echo "SETUP_DIR: $SETUP_DIR"
echo "DNS_NAME: $DNS_NAME"

WORK_DIR=$(pwd)

apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common cifs-utils tmux

### tmux completion
wget -O /etc/bash_completion.d/tmux https://raw.githubusercontent.com/imomaliev/tmux-bash-completion/master/completions/tmux

### jmespath command line parsing ###
wget https://github.com/jmespath/jp/releases/download/0.1.3/jp-linux-amd64 -O /usr/local/bin/jp \
        && sudo chmod +x /usr/local/bin/jp

### azure cli ###
# see: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest#install-with-apt-package-manager
echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ wheezy main" | \
     tee /etc/apt/sources.list.d/azure-cli.list
apt-key adv --keyserver packages.microsoft.com --recv-keys 52E16F86FEE04B979B07E28DB02C46DF417A0893
apt-get install apt-transport-https
apt-get update && sudo apt-get install azure-cli
# login with pcs-dev-ml-automation service principal
# https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest
az login --service-principal \
    --username [uname] --password "[pwd]!" \
    --tenant [tenant-uid]

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
# echo "//<storage-account-name>.file.core.windows.net/<share-name> /mymountpoint cifs vers=3.0,username=<storage-account-name>,password=<storage-account-key>,dir_mode=0777,file_mode=0777,serverino" >> /etc/fstab
AZ_STORAGE_NAME="[storage-name]"
AZ_STORAGE_KEY="[key]"
function _mount_az_file_share() {
    SHARE_NAME=$1
    MNT_BASE_DIR=$2

    mkdir -p $MNT_BASE_DIR/$SHARE_NAME

    FSTAB_RECORD="//${AZ_STORAGE_NAME}.file.core.windows.net/${SHARE_NAME} ${MNT_BASE_DIR}/${SHARE_NAME} cifs vers=3.0,username=${AZ_STORAGE_NAME},password=${AZ_STORAGE_KEY},dir_mode=0777,file_mode=0777,serverino"
    echo $FSTAB_RECORD >> /etc/fstab
}
_mount_az_file_share training-data /var/pcsml/azfiles
_mount_az_file_share remote-exec-out /var/pcsml/azfiles
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
_dkr_volume_create pcsml.remote-exec-out /var/pcsml/azfiles/remote-exec-out

# restart dockerd, just in case
systemctl daemon-reload
systemctl enable docker
systemctl restart docker

# reset home dir permissions
chown -R pcsdev:pcsdev ~/

### add the remote user to the docker group ###
usermod -a -G docker pcsdev