#!/bin/bash

# This script prepares kubespray by cloning the code and then removing unnecessary files so that this can be added in 
# the current repository and configured for further usage, etc.

set -e  # Exit on any error

# ================================= Step 1: Clone Kubespray =================================
git clone https://github.com/kubernetes-sigs/kubespray.git
# Go to fabric/kubespray (assuming the script is executed from the fabric directory)
cd kubespray

# ================================= Step 2: Checkout a stable release branch specific version =================================
# (this checks out the branch of kubespray called release-x)
git checkout release-2.27

# Step 3: Install Python dependencies (installs Ansible, etc.)
pip3 install -r requirements.txt

# ================================= Step 3: Clean up unnecessary files =================================
# (-rf is -r for recursive to delete folders and -f to force delete)
rm -rf .git # Git file with a lot of Git related files that are not necessary
rm -rf docs # Docs is not necessary, can be viewed here: https://github.com/kubernetes-sigs/kubespray
rm -rf .github .gitlab-ci # Git related files such as workflows, etc. not necessary 
rm -rf contrib # Contributing not necessary
rm -rf logo # Logo images not necessary
rm -rf scripts # Scripts not necessary
rm -rf test-infra tests # Tests not necessary
rm -rf extra_playbooks # Extra playbooks not necessary

# Delete specific files that are not necessary
# Git specific files
rm -f .gitattributes .gitignore .gitlab-ci.yml .gitmodules .pre-commit-config.yaml
# Remove lint and style files
rm -f .ansible-lint .ansible-lint-ignore .editorconfig .md_style.rb .mdlrc .nojekyll .yamllint
# License and other docs files not necessary
rm -f CHANGELOG.MD code-of-conduct.md CONTRIBUTING.md OWNERS OWNERS_ALIASES README.md RELEASE.MD LICENSE SECURITY_CONTACTS
# Other files
rm -f index.html

# ================================= Step 4: Replace symbolic links with their targets =================================
# This step is done after removing unnecessary files to avoid doing unnecessary additional operations of symlinks for removed files.

# Sybolic links can cause problems (this is used in Kubespray, such as: https://github.com/kubernetes-sigs/kubespray/blob/release-2.27/extra_playbooks/inventory)
# It is used to refer to files, mainly in Linux environments, to avoid copying them. This resulted in errors, such as:
# error: open("fabric/kubespray/extra_playbooks/inventory"): Invalid argument 
# error: unable to index file 'fabric/kubespray/extra_playbooks/inventory' 
# fatal: adding files failed

# Therefore, we find them and replace them with their target files with these commands
# Enable symlink in git
git config --global core.symlinks true
# Find all symbolic links and replace them with their actual content
find . -type l | while read symlink; do
    target=$(readlink "$symlink")
    symlink_dir=$(dirname "$symlink")
    absolute_target=$(realpath "$symlink_dir/$target")

    echo "[INFO] $symlink → $target"
    echo "[INFO] Copying from $absolute_target to $symlink"

    rm "$symlink"

    if [ -d "$absolute_target" ]; then
        cp -a "$absolute_target" "$symlink"
        echo "Replaced directory symlink $symlink with directory copy of $absolute_target"
    else
        cp "$absolute_target" "$symlink"
        echo "Replaced file symlink $symlink with file copy of $absolute_target"
    fi
done

# ================================= Display result message =================================
echo "Kubespray prepared and cleaned. Now you can configure it and use it to manage and configure the Kubernetes cluster."
