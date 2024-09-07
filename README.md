# Syngenta_project
This is a code for syngenta. 


## Cuda installation check 
Make sure a cuda 11.* is installed in your computer. 

To check which version of cuda you have simply type the following on a terminal 

```bash
nvcc --version
```

Ensure that the path and library for CUDA are set in your .bashrc file:

```bash
export PATH="$HOME/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```
## Installation

Installation steps:
* git clone, or download, this repository in a folder of your computer
* git clone https://github.com/WeheliyeHashi/Syngenta_project.git
* open a terminal window in that folder


```bash
conda env create -f requirements.yml
conda activate DrosP_code
pip install -e .
```


## Starting the program

Now that the code is installed, you can check if the installation was successful by typing the following in the terminal:

```bash
process_data --help
```
Make sure the DrosP_code environment is active.

### Updating an existing installation

Assuming that this code was cloned or downloaded to the desktop and that the DrosP_code environment has already been created, you can update the code by executing:

```bash
cd ~/Syngenta_project
conda activate DrosP_code
git pull
pip install -e .
```

## Project layout

Your project layout should look like this: It should include the RawVideos folder along with the Auxiliary file. An example of the metadata is also provided.

```
└── 📁Project_1
    └── 📁AuxiliaryFiles
        └── 📁CTRL - untreated
            └── 📁time 0h
                └── metadata_source.xlsx
           
    └── 📁RawVideos
        └── 📁CTRL - untreated
            └── 📁time 0h
```

## Processing data

Finally to process your data simply open a terminal:

```bash

conda activate DrosP_code
process_data --rw_path "path to your rawvideos folder"
```

This will create two new folders: one for each video containing results, and another for feature summaries and figures, which include boxplots and a clustermap for your data.

