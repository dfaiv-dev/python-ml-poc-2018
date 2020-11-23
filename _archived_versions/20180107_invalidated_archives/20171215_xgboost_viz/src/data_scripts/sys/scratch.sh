#!/usr/bin/env bash

restore_script="ubu1604-mbp-db-restore.sh"
scp ./$restore_script dfaivre@ubu1604-mbp:~/
ssh -t dfaivre@ubu1604-mbp "sudo bash ~/$restore_script hello!"