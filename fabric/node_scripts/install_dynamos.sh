#!/bin/bash

{
# Install Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
sudo apt-get install apt-transport-https --yes
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm

function addPathExport () {
  echo -e "export PATH=$1" >> $HOME/.bashrc
  echo "[inserted] export PATH=$1"
}

# Install CLI
curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install-edge | sh

# Add Linkerd to PATH
addPathExport "\$PATH:/home/ubuntu/.linkerd2/bin"
export PATH=$PATH:/home/ubuntu/.linkerd2/bin

kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml

linkerd check --pre                         # validate that Linkerd can be installed
linkerd install --set nodeSelector."kubernetes\\.io/hostname"=dynamos --crds | kubectl apply -f - # install the Linkerd CRDs
linkerd install --set nodeSelector."kubernetes\\.io/hostname"=dynamos | kubectl apply -f - # install the control plane into the 'linkerd' namespace
linkerd check                               # validate everything worked!

# Install Jaeger onto the cluster for observability
linkerd jaeger install \
  --set collector.nodeSelector."kubernetes\\.io/hostname"=dynamos \
  --set nodeSelector."kubernetes\\.io/hostname"=dynamos \
  --set jaeger.nodeSelector."kubernetes\\.io/hostname"=dynamos \
  --set webhook.nodeSelector."kubernetes\\.io/hostname"=dynamos \
  | kubectl apply -f -

mkdir homebrew
curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C homebrew

addPathExport "\$PATH:$HOME/homebrew/bin"
export PATH="$PATH:$HOME/homebrew/bin"

brew update
brew install etcd

echo -e "127.0.0.1 api-gateway.api-gateway.svc.cluster.local" | sudo tee -a /etc/hosts

git clone https://github.com/Javernus/DYNAMOS.git

cd DYNAMOS

./configuration/dynamos-configuration.sh

}