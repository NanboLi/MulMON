# MulMON

### (Pre-relsease)


This repository contains a PyTorch implementation of the [paper:  
**Learning Object-Centric Representations of Multi-object Scenes from Multiple Views**](https://github.com/NanboLi/MulMON)
**Li Nanbo**,
[**Cian Eastwood**](http://homepages.inf.ed.ac.uk/s1668298/),
[**Robert B. Fisher**](https://homepages.inf.ed.ac.uk/rbf/)  
NeurIPS 2020 (**<font style="color:red">Spotlight</font>**)  


## Working examples
<p float="left">
    <img src="assets/work1.gif" width="800"/>  
    <img src="assets/jaco_cam_traj_white.gif" width="800"/>  
    <img src="assets/dist1.gif" width="800"/>
</p>


## Requirements
**Hardware**:   
* GPU. Currently this code can only be run on GPU devices, however, we will consider adding a demo code that uses CPU only in the future.
* Disk space: we do NOT have any hard requirement for the disk space this is totally data dependent. See [Data](#data) section for more details.


**Python Environement**:  
1. We use Anaconda to manage our python environment. Check conda installation guide here: https://docs.anaconda.com/anaconda/install/linux/.

2. Open a new terminal, create a new conda env called "<font style="color:blue">mulmon</font>" and activate it:
```setup
conda env create -f ./conda-env-spec.yml  
conda activate mulmon
```

3. Install a gpu-supported PyTorch (tested with PyTorch 1.1, 1.2 and 1.7). See the installation instructions on
the PyTorch official site: https://pytorch.org/.

4. Install additional packages:
```setup
pip install tensorboard  
pip install scikit-image
```
If pytorch <=1.2 is used, you will also need to execute: ```pip install tensorboardX``` and import it in the `./trainer/base_trainer.py` file. This can be down by `commenting the 4th line` and `uncommenting the
5th line` in that file.


## Data
* **Get Dataset**  
  * **Easy way**:   
    [download](#) our datasets (available soon). We used only a subset of the DeepMind GQN-Jaco dataset, more available at [deepmind/gqn-datasets](https://github.com/deepmind/gqn-datasets).  
  * **Fun way**:   
    For multi-view CLEVR datasets, customisation is highly recommended (more fun!). An easy setup (follow the instructions in [CLEVR-MultiView repository](https://github.com/NanboLi/CLEVR-MultiView)) will allow you to customise scenes and generate your own multi-view CLEVR datasets.

* **Data structure** (important):  
Either way you go to get the dataset, the structure of your data should have a structure as follows:
```
<your path>
    ├── ...
    └── MultiObj
          ├── clevr                                   # place your own CLEVR-MV under this directory if you go the fun way
          │    ├── ...
          │    ├── clevr_mv            
          │    │    └── ... (omit)                    # see clevr_<xxx> for subdirectory details
          │    ├── clevr_aug           
          │    │    └── ... (omit)                    # see clevr_<xxx> for subdirectory details
          │    └── clevr_<xxx>
          │         ├── ...
          │         ├── data                          # contains a list of scene files
          │         │    ├── CLEVR_new_#.npy          # one .npy --> one scene sample
          │         │    ├── CLEVR_new_#.npy       
          │         │    └── ...
          │         ├── clevr_<xxx>_train.json        # meta information of the training scenes
          │         └── clevr_<xxx>_test.json         # meta information of the testing scenes  
          └── GQN  
               ├── ...
               └── gqn-jaco                 
                    ├── gqn_jaco_train.h5
                    └── gqn_jaco_test.h5
```



## Pre-trained models  
[Download](#) will be available soon.



## Usage
**Configure data path**:  
To run the code, you need to configure the data path in the scripts file first. For example,



* **Demo** (test run):  
TODO ...  

* **Train**  
* On a single gpu:  
```run
TODO ...  
```
* On multiple GPUs:  
```run
TODO ...  
```

* **Evaluation**  
TODO ...  



## Contact
We constantly check ''issues'', feel free to leave comments there. Or, email directly:  
nanbo.li@ed.ac.uk  


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
