These are some instructions I've written to follow how I have made flamedisx run on my M1 chip Macbook, as well as some notes I've found along the way.

Flamedisx works on M1 chips fine, the only issues with installation are that Tensorflow and anaconda aren't natively compatible with the ARM instruction set. Luckily apple appreciated the importance of machine learning and that frequently uses Tensorflow, so they have branched it and made a version compatible with M1 chips. Additionally miniforge3 can be used to set up Conda environments in place of Anaconda.

Lastly a word on pitfalls. Early after the release of M1 chips, apple branched Tensorflow 2.4.0 on GitHub, made it ARM compatible and released it. If you look for guides online most of them will point to this outdated GitHub branch and you don't want that. You actually just want the `tensorflow_macos` package.

### Installing Xcode

I have Xcode installed and it's presence is frequently assumed as a given, but it may not be necessary. Xcode is installed from the App Store and the command line tools are installed with `xcode-select --install`.

### Step 1: Install Miniforge3

Head over to [https://github.com/conda-forge/miniforge/releases](url) and download the latest version of Miniforge3 for macOS and with arm64 architecture. You're looking for a file with the naming convention `Miniforge3-VERSIONNUM-MacOSX-x86_64.sh`

Install it with 

`bash Miniforge3-VERSIONNUM-MacOSX-x86_64.sh`

then run `conda` to check it's working.

### Step 2: install Tensorflow

First create your conda environment with

`conda create --name flamedisxInstall `

Then activate it with

`conda activate flamedisxInstall `

where `flamedisxInstall`is the environment name, feel free to choose your own.

Then install the Tensorflow dependencies with 

`conda install -c apple tensorflow-deps`

and then Tensorflow itself with

`python -m pip install tensorflow-macos`

### Step 4: Dependencies and Flamedisx

Start by installing all of the various dependencies that flamedisx will require, plus Jupyter notebook and the package required to make it play nicely with Conda environments.

`conda install -c conda-forge notebook`
`conda install -c conda-forge nb_conda_kernels`
`conda install -c conda-forge pandas`
`conda install -c conda-forge numpy`
`conda install -c conda-forge tensorflow-probability`
`conda install -c conda-forge cloudpickle`
`conda install -c conda-forge tqdm`
`conda install -c conda-forge iminuit`
`pip install git+https://github.com/JelleAalbers/multihist.git`
`pip install git+https://github.com/JelleAalbers/wimprates.git`

Then find somewhere you want to put the git clone of flamedisx and download it.

`git clone https://github.com/FlamTeam/flamedisx.git`
`cd flamedisx`

Then check out the branch you want to run, here we get the master

`git checkout master`
`git pull origin master`

and then install it

`python setup.py develop`

### Step 5: Jupyter notebook

Run jupyter notebook with `jupyter notebook`

When creating a new notebook you should have the option of which kernel to run the notebook with, and that should include your conda environment, being labelled as `Python [conda env:flamedisxInstall]`where flamedisxInstall is the name of your environment. If you are in a notebook the kernel in use should be displayed in the upper right corner, and can be change by going to the dropdown menu: Kernel -> Change Kernel -> Python [conda env:flamedisxInstall]

You should now be able to import flamedisx with the usual

`import flamedisx as fd`
