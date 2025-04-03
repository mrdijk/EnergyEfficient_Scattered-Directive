# Setup Kubernetes
This file explains how to setup Kubernetes in FABRIC.

Note: this guide used WSL for the underlying testing and validation.

TODO: explain here how to do that.


## Configure Kubernetes environment
### Preparing Kubespary
In a Linux terminal (e.g. WSL), execute the following commands to prepare Kubespray for usage:
```sh
# Navigate to the fabric folder
cd fabric

# Use custom script to prepare kubespray, fetching the code from GitHub and cleaning up unnecessary files
./k8s/prepare_kubespray.sh
```
Now, when you want a new Kubespray version you can follow these steps again, but keep in mind that the local changes might get lost, so make a copy of those before doing this script.  Also, keep in mind that the script is specific for a version of Kubespray, files may be renamed or added, etc., so the script section removing files may need to be slightly altered when updating versions.

This seutp allows for easily configuring Kubespray with only the necessary files and saving changes in this GitHub repository for this project without interference or more manual steps to set it up in the future, etc.

### Using Kubespray
TODO: explain here to use the k8s-setup.ipynb notebook first.

In a Linux terminal (e.g. WSL), execute the following commands to use Kubespray to setup the Kubernetes cluster:
```sh
# Go into the kubespray directory
cd fabric/kubespray

# Set up your inventory for your cluster (will create files in fabric/kubespray/inventory/x)
cp -rfp inventory/sample inventory/dynamos-cluster
# Then add the inventory.ini file in the created dynamos-cluster folder. The k8s_setup.ipynb notebook gets the necessary information

# Configure the Ansible config file (by default it does not allow it in the working directory: https://docs.ansible.com/ansible/devel/reference_appendices/config.html#cfg-in-world-writable-dir)
# For example:
export ANSIBLE_CONFIG=/mnt/c/Users/cpoet/VSC_Projs/EnergyEfficiency_DYNAMOS/fabric/kubespray/ansible.cfg

# Then execute the playbook to configure the cluster, this takes a while to execute, the more nodes the longer it takes
ansible-playbook -i inventory/dynamos-cluster/inventory.ini cluster.yml -b -v --private-key=~/.ssh/fabric_bastion_key
# TODO: test this now and see the results and fix any problems
# TODO: this needs to be bastion key? See k8s-setup.ipynb
# TODO: now working on SSH access, and continue locally
# TODO: key required for SSH access is probably the slice_key OR something else from /fabric_config, such as fabric_bastion_key, test that!

```
