# MulMON

This repository contains a PyTorch implementation of the [paper:  
**Learning Object-Centric Representations of Multi-object Scenes from Multiple Views**](https://github.com/NanboLi/MulMON) (to appear)  
**Li Nanbo**,
[**Cian Eastwood**](http://homepages.inf.ed.ac.uk/s1668298/),
[**Robert B. Fisher**](https://homepages.inf.ed.ac.uk/rbf/)  
NeurIPS 2020 (**<font style="color:red">Spotlight</font>**)  


## Pre-release Working Examples
<p float="left">
    <img src="assets/work1.gif" width="800"/>  
    <img src="assets/jaco_cam_traj_white.gif" width="800"/>  
    <img src="assets/dist1.gif" width="800"/>
</p>


## Requirements
**Hardware**:   
1. GPU. Currently this code can only be run on GPU devices, however, we will consider adding a demo code that uses CPU only in the future.
2. Disk space: we don't have any hard requirement for the disk space this is totally data dependent. See [Data](#data) section for more details.


**Python Environement**:  
1. We use Anaconda to manage our python environment. Check conda installation guide here: https://docs.anaconda.com/anaconda/install/linux/.

2. Create a new conda env called "<font style="color:blue">mulmon</font>" and activate it:
```setup
conda env create -f ./conda-env-spec.yml  
conda activate mulmon
```

3. Install a gpu-supported PyTorch (tested with PyTorch 1.1 and 1.7). See the installation instructions on
the PyTorch official site: https://pytorch.org/.

4. Install additional packages:
```setup
pip install tensorboard  
pip install scikit-image
```
If pytorch 1.1 is used, you will also need to execute: ```pip install tensorboardX``` and import it in the `./trainer/base_trainer.py` file. This can be down by `commenting the 4th line` and `uncommenting the
5th line` in that file.


## Data



# <font style="color:green">To finish</font>  


## Cite
Please cite the paper if you find the code useful:
```latex
@inproceedings{nanbo2020mulmon,
      title={Learning Object-Centric Representations of Multi-Object Scenes from Multiple Views},
      author={Nanbo, Li and Eastwood, Cian, and Fisher, Robert B},
      year={2020},
      booktitle={Advances in Neural Information Processing Systems},
}
```
