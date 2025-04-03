# FABRIC
For the deployment of DYNAMOS we use FABRIC: https://portal.fabric-testbed.net/

Useful article: https://learn.fabric-testbed.net/knowledge-base/things-to-know-when-using-fabric-for-the-first-time/


# FABRIC Notebooks
FABRIC recommends using Jupyter Notebook for interacting with FABRIC, which is where this folder comes into play. This folder contains the notebooks that can be used for DYNAMOS in FABRIC specifically. Managing these notebooks in version control allows for easy management and version histories, and the notebooks can be edited in an editor such as VSC with the corresponding Jupyter extension (https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). These notebooks can be added in the Jupyter Hub in FABRIC: https://learn.fabric-testbed.net/article-categories/jupyter-hub/.

## Sequence of notebooks
TODO: add here the sequence of notebooks:

1. Configure and validate the FABRIC environment and project to be used: [configure_and_validate.ipynb](./configure_and_validate.ipynb)
2. Create a slice with the nodes to be used: [create_slice.ipynb](./create_slice.ipynb)
3. TODO: specific Kubernetes notebook to setup the cluster, including DYNAMOS setup and energy efficiency setups, etc.
TODO: next is probably to configure the nodes in other notebooks.