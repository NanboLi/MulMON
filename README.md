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

2. Open a new terminal, direct to the MulMON directory:
```path
cd <YOUR-PATH-TO-MulMON>/MulMON/
```
create a new conda environment called "<font style="color:blue">mulmon</font>" and then activate it:
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
If pytorch <=1.2 is used, you will also need to execute: ```pip install tensorboardX``` and import it in the `./trainer/base_trainer.py` file. This can be down by commenting the **4th line** and uncommenting the **5th line** of that file.


## Data
* **Data structure** (*important*):  
  We use a data structure as follow.
  ```
  <YOUR-PATH>                                          
      ├── ...
      └── mulmon_datasets
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
  We recommend one to get the necessary data folders ready before downloading/generating the data files:
  ```prepare
  mkdir <YOUR-PATH>/mulmon_datasets  
  mkdir <YOUR-PATH>/mulmon_datasets/clevr  
  mkdir <YOUR-PATH>/mulmon_datasets/GQN
  ```

* **Get Datasets**  
  * **Easy way**:  
    Download our datasets:  
      * [clevr_mv.tar.gz](https://drive.google.com/file/d/1P8Upanh7yan3kKdnNAtXpa6uC1819hUO/view?usp=sharing) and place it under the `<YOUR-PATH>/mulmon_datasets/clevr/` directory.  
      * [clevr_aug.tar.gz](https://drive.google.com/file/d/1dL_YeGSINghTTHoOd8gjbSMm2u5CagJa/view?usp=sharing) and place it under the `<YOUR-PATH>/mulmon_datasets/clevr/` directory.  
      * [gqn-jaco.tar.gz](https://drive.google.com/file/d/1xDm0InpO6jLILbOFN5MhaoJvjenDzuaD/view?usp=sharing) and place it under the `<YOUR-PATH>/mulmon_datasets/clevr/` directory.  

    and extract them in places. For example, the command for extracting `clevr_mv.tar.gz`:
    ```
    tar -zxvf <YOUR-PATH>/mulmon_datasets/clevr/clevr_mv.tar.gz
    ```
    Note that: 1) we used only a subset of the DeepMind GQN-Jaco dataset, more available at [deepmind/gqn-datasets](https://github.com/deepmind/gqn-datasets), and 2) the published clevr_aug dataset differs slightly from the one we used in the paper.

  * **Fun way**  
   (**available soon...**):  
    <!-- For multi-view CLEVR datasets, customisation is highly recommended (more fun!). An easy setup (follow the instructions in [CLEVR-MultiView repository](https://github.com/NanboLi/CLEVR-MultiView) will allow you to customise scenes and generate your own multi-view CLEVR datasets. -->


## Pre-trained models  
[Download the pretrained models](https://drive.google.com/file/d/1CBCGftjIsZhrw0KIkNNTJJeexUulPDdn/view?usp=sharing) (&leftarrow; click) and place it under `MulMON/', i.e. the root directory of this repository, then extract it by executing: ```tar -zxvf ./logs.tar.gz```

## Usage
**Configure data path**  
To run the code, the data path, i.e. the `<YOUR-PATH>` in a script, needs to be correctly configured. For example, we store the MulMON dataset folder `mulmon_datasets` in `../myDatasets/`, to train a MulMON on GQN-Jaco dataset using a single GPU, the **4th line** of the `./scripts/train_jaco.sh` script should look like:
`data_path=../myDatasets/mulmon_datasets/GQN`.

* **Demo**    
  (**available soon...**)
  <!-- *Dowload the pre-trained models first* and execute:
  ```run
  . scripts/demo.sh  
  ```
  Check `./logs` folder for the generated demos. -->

* **Train**  
  * On a single gpu (e.g. using the GQN-Jaco dataset):  
  ```run
  . scripts/train_jaco.sh  
  ```
  * On multiple GPUs (e.g. using the GQN-Jaco dataset):  
  ```run
  . scripts/train_jaco_parallel.sh  
  ```  
  * To resume weights from a pre-trained model, e.g. `checkpoint-epoch<#number>.pth`, simply append a flag `--resume_epoch <#number>` to one of the flags in the script files.  
  For example, to resume the weights from `checkpoint-epoch2000.pth` (trained on GQN-Jaco) for further training, the **10th line** of the `./scripts/train_jaco.sh` script should be configured as:  
  `--input_dir ${data_path} --output_dir ${log_path} --resume_epoch 2000 \`.


* **Evaluation**  
(**available soon...**)


## Contact
We constantly respond to the raised ''issues'' in terms of running the code. For further inquiries and discussions (e.g. questions about the paper), email: `nanbo.li@ed.ac.uk`.


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
