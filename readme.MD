# Scattered Directive

Welcome to Scattered Directive. This is an automated and reproducible framework to deploy the digital data marketplace
platform [DYNAMOS](https://github.com/Jorrit05/DYNAMOS) on the [FABRIC](https://portal.fabric-testbed.net) Research
Infrastructure.

Scattered Directive automates the entire setup, from requesting nodes to configuring and deploying DYNAMOS with the
relevant relations to the nodes all defined. This allows researchers and developers (you!) to utilise DYNAMOS without
any hassle.

To prove our implementation of Scattered Directive, we utilised it to implement a Vertical Federated Learning workload,
creating a new archetype to support shared execution of the global model training on a trusted third party. This
archetype is aptly named *Shared Execution via TTP*.

## Setup

Practically everything is automated. The only thing you need to do is open the FABRIC Jupyter Notebook, clone this
repository into there, open `fabric/dynamos.ipynb` configure the preferred configuration of nodes and execute.
