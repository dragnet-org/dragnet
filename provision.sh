
# die on error
set -e

sudo apt-get update

sudo apt-get -y install libatlas-base-dev libatlas-dev lib{blas,lapack}-dev
sudo apt-get -y install libxslt-dev libxml2-dev gcc g++

# According to `conda list`, pip installing conda isn't the way to go. Instead, use this:
curl -O http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh
export PATH=$HOME/miniconda3/bin:$PATH
conda_deps='pip numpy scipy'
conda create -m -p $HOME/py --yes $conda_deps python=2.7
export PATH=$HOME/py/bin:$PATH

# configure conda for future login
echo "export PATH=$PATH" >> $HOME/.bashrc

pip install "Cython>=0.21.1"
pip install -r requirements.txt

make install

