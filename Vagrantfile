# -*- mode: ruby -*-
# vi: set ft=ruby :

ENV['VAGRANT_DEFAULT_PROVIDER'] = 'virtualbox'

Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/trusty64"

    # lxml has trouble building if the amount of memory is 512:
    # http://stackoverflow.com/questions/16149613/installing-lxml-with-pip-in-virtualenv-ubuntu-12-10-error-command-gcc-failed
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "1024"
  end

   config.vm.provision "shell", privileged: false, inline: <<-SHELL
     cd /vagrant; ./provision.sh
   SHELL
end
