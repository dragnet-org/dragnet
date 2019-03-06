FROM ubuntu:14.04
WORKDIR /vagrant
COPY . /vagrant

RUN ./provision.sh