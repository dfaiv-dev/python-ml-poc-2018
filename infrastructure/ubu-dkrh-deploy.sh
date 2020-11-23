#!/bin/bash

# should rewrite with getopt, getopts or powershell?

REMOTE=$1
INFRASTRUCTURE_PATH=$2
REMOTE_USER="pcsdev"

echo "REMOTE (0): $REMOTE, INFRASTRUCTURE_PATH (1): $INFRASTRUCTURE_PATH"


scp -r ${INFRASTRUCTURE_PATH}/ubu-dkr-setup $REMOTE_USER@$REMOTE:~/

REMOTE_SETUP_CMD="sudo bash ~/ubu-dkr-setup/setup.sh \
~/ubu-dkr-setup \
$REMOTE"

ssh -t $REMOTE_USER@$REMOTE $REMOTE_SETUP_CMD

# same idea for remote-tls-setup.sh