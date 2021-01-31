# Unerwater Image Dataset and Image Classifier

This is a complete set of an initial attempt to create a basic dataset containing 2 classes of underwater images, a class of pipes and a class with the underwater environment generally.

* The new dataset can be found [here](https://drive.google.com/file/d/1zsMznVwrxo1gUF-1sQYIh0UID05IDcSS/view?usp=sharing)

# Anaconda configuration for CPU and GPU

Deploying a Machine Learning and Deep Learning poject is not always straight forward particularly when we need to run the model in the GPU, if there is one in our machine.
First of all, we need to have a GPU that suports the CUDA library. The list of the CUDA supported GPUs can be found [here](https://developer.nvidia.com/cuda-GPUs).
The next stpep is to specify which framework we intend to use for our deployment, it will be TensorFlow, Pytorch, MXNET etc, for this example we will use the TensorFlow.

Before installing anything it is the best practice to use virtual environments to install the specific versions that we want, and keep the main system intact, otherwise by installing different versions for different projects we will face mant difficulties with the conflicted libries. Here we'll use Anaconda to setup our virtual environments.

## Installing Anaconda

The cool thing using Anaconda is that allows as to install the required from the GPU drivers and not bother to install the drivers locally to the machine, which could be rather a messy procedure. The other best thing when using Anaconda environments is that we can have differnt environments with different CUDA and TensorFlow versions. Also by installing Anaconda Python is included.

I am using the Miniconda version which is much lighter than the full Anaconda version, and the linux version can be downloaded from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh).
After the isntallation has finished we need to exit the terminal and open it again in order the installation of Anaconda to take effect.

Before we steup the environment need to find the correct versions of CUDA, cuDNN, etc required by [TensorFlow](https://www.tensorflow.org/install/gpu).

## TensorFlow Requirements
To istall **TensorFlow 2.4** and **TensorFlow 2.4 GPU** we need:
1. Python 3.5 - 3.8
2. For Linux, Ubuntu 16.04 or later (we can use other Ubuntu based distros as well)
3. [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) —CUDA® 11.0 requires 450.x or higher.
4. [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) —TensorFlow supports CUDA® 11 (TensorFlow >= 2.4.0)
5. [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA® Toolkit.
6. [cuDNN SDK 8.0.4](https://developer.nvidia.com/cudnn) [cuDNN versions](https://developer.nvidia.com/rdp/cudnn-archive).
7. (Optional) [TensorRT 6.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) to improve latency and throughput for inference on some models.

## Search Anaconda repos for the needed packages
Since, we'll use conda environments to install all the necessary drivers, only we need to ensure that the machine has the corect Nvidia GPU drivers (450.x or higher). Then we need to make check Anaconda ropsitories to find if the above drivers existing. Fortunatly, we can make use of the conda-forge channel, as well as the nvidia channel to install these drivers.
We can do that as follows:
```zsh
conda search cuda

Loading channels: done
No match found for: cuda. Search: *cuda*
# Name                       Version           Build  Channel             
cudatoolkit                      9.0      h13b8566_0  pkgs/main           
cudatoolkit                      9.2               0  pkgs/main           
cudatoolkit                 10.0.130               0  pkgs/main           
cudatoolkit                 10.1.168               0  pkgs/main           
cudatoolkit                 10.1.243      h6bb024c_0  pkgs/main           
cudatoolkit                  10.2.89      hfd86e86_0  pkgs/main           
cudatoolkit                  10.2.89      hfd86e86_1  pkgs/main           
cudatoolkit                 11.0.221      h6bb024c_0  pkgs/main
```
For cudnn will use the nvidia chanel:

```zsh
conda search -c nvidia cudnn

Loading channels: done
# Name                       Version           Build  Channel                       
cudnn                          8.0.0      cuda10.2_0  nvidia              
cudnn                          8.0.0      cuda11.0_0  nvidia              
cudnn                          8.0.4      cuda10.1_0  nvidia              
cudnn                          8.0.4      cuda10.2_0  nvidia              
cudnn                          8.0.4      cuda11.0_0  nvidia              
cudnn                          8.0.4      cuda11.1_0  nvidia
```
So, for TF-2.4 we'll install:
1. Python 3.8
2. cudatoolkit 11.0
3. cudnn 8.0.4
4. tensorflow-gpu 2.4

Now that we have found the correct versions of the necessary libraries and drivers for the TF-2.4, the next step is to create the virtual environment, as the official Anaconda [documentation](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/) describes, that will host the packages and libraries required by TensorFlow with the following command:

```zsh
conda create -n tf24-cuda11 python=3.8
```

After crating the `tf24-cuda11` environment, we install the above packages plus the jupyter lab:

```zsh
conda install cudatoolkit

conda install -c nvidia cudnn=8

pip install tensorflow-gpu
```
The `pip install tensorflow-gpu` commande will install Tensorflow gpu version, Tensorflow estimator, Tensorflow base. We don't use conda to install tensorflow-gpu because the latest tensorflow version in conda repo is 2.2 for Linux and 2.3 for Windows. If we do use `conda install tnsorflow-gpu`, it will install also CUDA 10.2 and cuDNN 7.

The [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) can be install from the conda-forge channel:
```zsh
conda install -c conda-forge jupyterlab
```
In the new version of jupyter lab 3.x the `tab` completion does not work, and we need to downgrade the `jedi` library to 0.17.2. So, actually we install: 

```zsh
conda install -c conda-forge jupyterlab

conda install -c conda-forge jedi=0.17.2
```

Additionally, I have created a `requiremets.txt` to install some extra frameworls and libraries such as:
1. Matplotlib
2. OpenCV
3. Scikit-learn
4. Pillow, etc

with the following command: `pip install -r requiremets.txt`

## Register the environment
It is also a good practice to register the ipykernel for the given environment. After activating the conda environemt, in this case is `conda activate tf24-cuda11` we use the following command to [link](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) the kernel with the environment using `ipykernel install --user --name myenv --display-name "my_environment_name"`:
```zsh
ipykernel install --user --name tf24-cuda11 --display-name "Python 3.8.5 (tf24-cuda11)"
```

## Test the environment
After completing the above steps we can test the installation.

```python
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(logical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

Tensor Flow Version: 2.4.1
Keras Version: 2.4.0

Python 3.8.5 (default, Sep  4 2020, 07:30:14) 
[GCC 7.3.0]
Pandas 1.2.1
Scikit-Learn 0.24.1
1 Physical GPUs, 1 Logical GPUs
Num GPUs Available:  1
```
## Managint the Environment

So far we have created the environment for TensorFlow 2.4 with the appropriate CUDA and cuDNN drivers installed through conda repos. Also, to make our life easier, the different libraries installed using the `requirements.txt` file. Now, it's a good opportunity to export the environment by creating a `.yml` file using the command:
```zsh
conda env export > tf24-cuda11.yml
```
Finallay, with this way, of using Anaconda environments to install CUDA drivers, we can install multiple versions of TensorFlow as well as of other ML/DL frameworks, without mess the systems libraryries. Also, another way is to use Docker containers, more details can be found:
* [In TensorFlow's installation guide](https://www.tensorflow.org/install)
* [In this Medium article](https://blog.kovalevskyi.com/multiple-version-of-cuda-libraries-on-the-same-machine-b9502d50ae77)
* [Managing CUDA dependencies with Conda](https://towardsdatascience.com/managing-cuda-dependencies-with-conda-89c5d817e7e1)
