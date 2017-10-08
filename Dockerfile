FROM ubuntu
MAINTAINER Raphael Osorio<raphaelaosorio@gmail.com>

# Create the directory where we will copy the application code
RUN mkdir /srv/app
# ... and declare it as the default working directory
WORKDIR /srv/app

# Install required software:
RUN apt-get update
RUN apt-get -y install --upgrade python python-pip python-dev build-essential python-setuptools build-essential  && \
    apt-get -y install apt-utils && \
    apt-get -y install libatlas-base-dev libatlas-dev && \
    apt-get -y install libblas-dev liblapack-dev && \
    apt-get -y install libxslt-dev libxml2-dev gcc g++ && \
    apt-get -y install curl && \
    apt-get -y install bzip2


# Create runtime environment
RUN curl -O https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
RUN bash Miniconda2-latest-Linux-x86_64.sh -b
RUN rm Miniconda2-latest-Linux-x86_64.sh
RUN conda_deps='pip numpy scipy'
RUN /root/miniconda2/bin/conda create -m -p $HOME/py --yes $conda_deps python=2.7

# Copy as early as possible so we can cache ...
# DIDN'T WORK
#ADD ./requirements.txt /srv/app/requirements.txt


# Install application dependencies
RUN pip install "Cython>=0.21.1" && \
    pip install --upgrade pip && \
    pip install nose && \
    pip install coverage && \
    pip install Cython>=0.21.1 && \
    pip install lxml && \
    pip install scikit-learn>=0.15.2 && \
    pip install numpy && \
    pip install scipy && \
    pip install mozsci

# Add the application
ADD . /srv/app


# activate environment, install python dependencies, and run make install
RUN export PATH=$HOME/miniconda2/bin:$PATH && \
    /bin/bash -c "source activate /root/py/ && pip install 'Cython>=0.21.1' && pip install nose && pip install coverage && pip install lxml && pip install scikit-learn==0.18.1 && pip install numpy && pip install scipy && pip install mozsci && make install"

# HOW TO RUN
#docker container run -it dragnet-1 /bin/bash -c "source /root/miniconda2/bin/activate /root/py && python my_script.py"