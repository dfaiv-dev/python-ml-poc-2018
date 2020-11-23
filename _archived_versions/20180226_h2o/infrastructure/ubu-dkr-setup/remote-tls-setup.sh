#!/usr/bin/env bash

# enable remote docker api over TLS
# must be run for each VM/DNS name until someone tells me how
#   to create a server-cert for ALL DNS names.

SETUP_DIR=$1
CERT_DIR=$2
DNS_NAME=$3
WORK_DIR=$(pwd)

# docker tls cert deploy
mkdir -p /var/docker

# make a new cert, signed by our ca, for the host
# see: https://docs.docker.com/engine/security/https/#create-a-ca-server-and-client-keys-with-openssl

# make a tmp working dir
# copy in the CA cert and key
# TODO: remove sensitive info -- ca key password: [pwd]
rm -rf ~/tmp
mkdir ~/tmp
cp $CERT_DIR/ca-key.pem ~/tmp
cp $CERT_DIR/ca.pem ~/tmp
cd ~/tmp
echo "generating server certs for $DNS_NAME"
openssl genrsa -out server-key.pem 4096
openssl req -subj "/CN=$DNS_NAME" -sha256 -new -key server-key.pem -out server.csr

# I'm pretty sure we only need to use the $DNS_NAME, and 127.0.0.1 IP,
# but throwing some other stuff at it as well
# DMF 2017-12-13
HOST=$(hostname)
echo subjectAltName = DNS:$HOST,DNS:$DNS_NAME,DNS:*,DNS:*.com,IP:0.0.0.0,IP:127.0.0.1 >> extfile.cnf
echo extendedKeyUsage = serverAuth >> extfile.cnf
openssl x509 -req -days 365 -sha256 -in server.csr -CA $CERT_DIR/ca.pem -CAkey $CERT_DIR/ca-key.pem \
  -CAcreateserial -out server-cert.pem -extfile extfile.cnf

cd $WORK_DIR
cp $CERT_DIR/ca.pem /var/docker
cp ~/tmp/server-cert.pem /var/docker
cp ~/tmp/server-key.pem /var/docker

# todo: add systemd docker.service override.conf file
cp $SETUP_DIR/docker-remote-daemon.json /etc/docker/daemon.json

# restart dockerd
systemctl daemon-reload
systemctl enable docker
systemctl restart docker