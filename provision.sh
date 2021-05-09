# die on error
set -e

apt-get update

export DEBIAN_FRONTEND=noninteractive

apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl \
  libxslt-dev libxml2-dev

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh
hash -r
conda_deps='pip numpy scipy'
conda create -m -p $HOME/py --yes $conda_deps python=3.7

pip install "Cython>=0.21.1"

chown -R dragnet:dragnet $HOME

# Clean up after installing packages to keep docker image smaller
apt-get clean
rm -rf /var/lib/apt/lists/*
