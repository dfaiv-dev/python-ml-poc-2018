#!/usr/bin/env bash

# deploy from local with:
# scp ubu1604-dhost-setup.sh ubu1604-dhost:~/
# ssh -t ubu1604-dhost "sudo bash ~/ubu1604-dhost-setup.sh"

apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# jmespath command line parsing
wget https://github.com/jmespath/jp/releases/download/0.1.3/jp-linux-amd64 -O /usr/local/bin/jp \
        && sudo chmod +x /usr/local/bin/jp

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

apt-key fingerprint 0EBFCD88

add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

apt-get update
apt-get install -y docker-ce

# setup dockerd
sed -i \
	s"@\(ExecStart=.*\)@ExecStart=/usr/bin/dockerd -H fd:// -H tcp://0.0.0.0:2375@" \
	/lib/systemd/system/docker.service
systemctl daemon-reload
systemctl enable docker
systemctl restart docker
