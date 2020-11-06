## Code for ICML 2020 submission: Occlusion-Free Multi-Object Representation Learning with Bayesian Surprise Suppression

*NOTE: ensure >=30GB free space for running the code. To reproduce the results in the paper, 2 GPUs (each has >=11GB memory) is the minimum requirement for training. However, the evaluation code supports single GPU only.*

#### A. Configure Python Environment  
1. We use Anaconda to manage our python environment, to use the code, one needs to install conda first. Download Anaconda from: https://www.anaconda.com/distribution/, and check installation guide in https://docs.anaconda.com/anaconda/install/linux/.
2. Once Anaconda is installed, please create a virtual environment with the provided __OccBSS.yml__ file. Open a terminal and execute: ```conda env create -f OccBSS.yml```. Then, activate the environment by executing: ```conda activate OccBSS``` in the same terminal.
3. If step 2 is successful, the activated __OccBSS__ virtual environment should now contains all but two packages needed for our code: PyTorch (>1.0) and TensorBoardX, we need to install them manually.
  * __PyTorch installation__: details refer to https://pytorch.org/get-started/locally/ (install the gpu-version pytorch with torchvision).  
  * __TensorBoardX installation__: in the activated __OccBSS__, execute ```pip install tensorboardX```.

#### B. Download Data
Download the three datasets used in the paper from: https://figshare.com/s/e8ead3808e2c5baebbc3, and place it under __./dataset/__ folder. Within the exact folder, extract all .zip files in places.

#### C. Configure the Code
Some modules used in our code are adapted from an open source repo https://github.com/applied-ai-lab/genesis. To avoid copyright issues, we do not include directly the modules in the submitted code but, alternatively, provide patches of our modifications to the original. Sorry for the inconvenience.

To configure our code, one needs to:
1. Pull the following files from the original repo first and place them all under our __./models/__ folder (where their patches are).
  * *./models/monet_config.py*
  * *./modules/blocks.py*
  * *./modules/component_vae.py*
  * *./modules/seq_att.py*
  * *./modules/unet.py*
2. In the opened terminal, execute the below commands to apply the modifications we added to the original modules:  
  ```
  patch ./models/monet_config.py ./models/monet_config.patch && mv ./models/monet_config.py ./models/baseline_monet.py    
  patch ./models/blocks.py ./models/blocks.patch  
  patch ./models/component_vae.py ./models/component_vae.patch
  patch ./models/seq_att.py ./models/seq_att.patch
  patch ./models/unet.py ./models/unet.patch
  ```
to remove all the patches after applying them, execute: ```rm -f ./models/*.patch```. However, we recommend to keep these patches.

#### D. Running Monet & MONO-BSS
If the steps listed above are successfully configured, one should be able to run our code now. We highly recommend one to customize the provided script __./train.sh__ to run the training code, and __./eval.sh__ to run the evaluation code. *(Reminder: make a copy of the .sh files before modifying them.)*

The training code will create a log folder __./logs/<*log_modelX*>/__ to store weights and model visualizations, where *modelX* is specified by the __--arch__ flag. The evaluation code will print all a result table in the terminal, and also generate a .json file to store all the results in the __./logs/<*log_modelX*>/__ folder. If __--vis_batch__ flag is specified, then visualization samples will be saved to __./logs/<*log_modelX*>/generated/__. Details about all the configurable flags used in the aforementioned .sh files refer to __./train_parallel.py__ and __./eval.py__ files.

Command for running a bash file in terminal (remember to check first if Conda environment __OccBSS__ is activated):```. ./xxxx.sh```, where *xxxx.sh* can be either *train.sh* or *eval.sh*.
(*Note: we provide also four trained models within the ./logs/ folder in case one wants to run evaluation code directly. For MDS data, although models of 1000-epoch training was used to produce the results in the main paper, longer training should achieve better resutls.*)
