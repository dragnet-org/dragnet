
# die on error
set -e

sudo apt-get update

sudo apt-get -y install libatlas-base-dev libatlas-dev lib{blas,lapack}-dev
sudo apt-get -y install libxslt-dev libxml2-dev gcc g++

curl -O https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh -b
rm Miniconda2-latest-Linux-x86_64.sh
export PATH=$HOME/miniconda2/bin:$PATH
conda_deps='pip numpy scipy'
conda create -m -p $HOME/py --yes $conda_deps python=2.7
export PATH=$HOME/py/bin:$PATH

# configure conda for future login (for vagrant)
echo "export PATH=$PATH" >> $HOME/.bashrc

pip install "Cython>=0.21.1"
pip install -r requirements.txt

make install

