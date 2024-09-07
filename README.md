# Syngenta_project
This is a code for syngenta. 


##Cuda installation check 
Make sure a cuda 11.* is installed in your computer. 

To check which version of cuda you have simply type the follwoing on a terminal 

```bash
nvcc --version
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

Now that the code is installed, you can check if installation was done correctly by typing this on the terminal
`process_data --help` in your terminal window (provided the `DrosP_code`
environment is active)


### Updating an existing installation

Assuming that this code was cloned or donwloaded to desktop and that the `Worm_annotation` environment has already been created, you can update the code by executing
```bash
cd ~/Syngenta_project
conda activate DrosP_code
git pull
pip install -e .
```




your_project/
│
├── RawVideos/
│   ├── compound_1
│   └── compound_2
│
└── AuxiliaryFiles/
    ├── compound_1 ── metadata_file.xls
    └── compound_2 ── metadata_file.xls



