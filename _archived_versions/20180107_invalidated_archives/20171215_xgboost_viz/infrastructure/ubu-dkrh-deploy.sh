#!/bin/bash

# should rewrite with getopt, getopts or powershell?

REMOTE=$1
SETUP_PATH=$2
SETUP_DIR_NAME=$(basename $SETUP_PATH)
REMOTE_USER="pcsdev"

echo "REMOTE (0): $REMOTE, SETUP FILES PATH (1): $SETUP_PATH"
echo "SETUP DIR NAME: $SETUP_DIR_NAME"


scp -r $SETUP_PATH $REMOTE_USER@$REMOTE:~/

REMOTE_SETUP_CMD="sudo bash ~/$SETUP_DIR_NAME/setup.sh \
~/$SETUP_DIR_NAME \
$REMOTE"

ssh -t $REMOTE_USER@$REMOTE $REMOTE_SETUP_CMD

# same idea for remote-tls-setup.sh