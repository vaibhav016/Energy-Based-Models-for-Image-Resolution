# Energy-Based Models for Image resolution

This project aims at image resolution using Energy-Based Models. 


-----------------------------------------------------------------------------------------------------------------------
## Requirements
The current version of the code has been tested with:
* `pytorch '1.12.1'`
* `torchvision 0.13.1`
-------------------------------------------------------------------------------
## Dataset:

For setting up the pipeline, currently the dataset used is CelebFaces LFW. 

http://vis-www.cs.umass.edu/lfw/lfw.tgz

--------------------------------------------------------------------------
## Some great resources for EBMS 

* https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

* https://atcold.github.io/pytorch-Deep-Learning/en/week13/13-1/

* http://www.cs.toronto.edu/~vnair/ciar/lecun1.pdf

* https://openai.com/blog/energy-based-models/

* https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch 

In the jupyter notebook, i have attributed the resources so that it becomes clear the source of code.

------------------------------------------------------------------------------

## Steps to run the code. 
1) Download the dataset and keep it in the same directory
2) Install torch and torchvision 
3) Install pytorch-lighning ` !pip install torch torchvision pytorch-lightning `
4) Make sure the versions are latest
5) Run the jupyter notebook (ideally it should run without any error, let me know)
------------------------------------------------------------------------
## Current status
1) The pipeline is working, but i havent checked the math out, we still need to investigate it. 
2) The model, the way its structured is very rudimentary and raw, nothing is optimal. 
3) In case we change this idea, we can follow a similar pipeline with relevant changes of components.