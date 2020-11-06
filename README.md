# MulMON

This repository contains a PyTorch implementation of the [paper:  
**Learning Object-Centric Representations of Multi-object Scenes from Multiple Views**](https://github.com/NanboLi/MulMON) (to appear)  
**Li Nanbo**,
[**Cian Eastwood**](http://homepages.inf.ed.ac.uk/s1668298/),
[**Robert B. Fisher**](https://homepages.inf.ed.ac.uk/rbf/)  
NeurIPS 2020 (**<span style="color:red">Spotlight</span>**)  


## Pre-release Working Examples
<p float="left">
    <img src="assets/work1.gif" width="800"/>  
    <img src="assets/jaco_cam_traj_white.gif" width="800"/>  
    <img src="assets/dist1.gif" width="800"/>
</p>


## Requirements
**Hardware**:   
1. GPU. Currently this code can only be run on GPU devices, however, we will consider adding a demo code that uses CPU only in the future.
2. Disk space:


**Python Environement**:  
1. We use Anaconda to manage our python environment. Check conda installation guide here: https://docs.anaconda.com/anaconda/install/linux/.

2. Create a new conda env called "<span style="color:blue">mulmon</span>" and activate it:
```setup
conda env create -f ./conda-env-spec.yml  
conda activate mulmon
```

3. Install a gpu-supported pytorch (tested with 1.1, 1.2 and 1.7).  
<span style="color:green">To finish ... </span>



4. Install additional packages:
```setup
pip install tensorboard  
pip install scikit-image
```
if you are using pytorch 1.1, you will also need to execute: ```pip install tensorboardX```.  
   



<span style="color:green">#To finish</span>
