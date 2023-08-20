# Time Series Ensembling ⏲️
This repository is meant to host the code and analysis for the project specified in this repository: https://github.com/DSML-Research-Group/public-projects/issues/2

The "big idea" behind the project is that combining different window sizes is a novel way of improving time series results, and allows models to improve both their bias and variance, since different window lengths contain both fundamentally different types of information about a time series, as well as serve the purpose of cancelling each other out.  

# Background Information 📖
Luckily for us, this problem is fairly novel, and represents a new way of looking at the domain of time series modeling.  However, the once precedent for using this technique and getting results with it was established in the N-Beats paper:  https://arxiv.org/abs/1905.10437

Ironically, this technique was NOT the main idea behind the paper, but they tried it and it worked.

# Helpful Links 🔗
 - N-Beats paper: https://arxiv.org/abs/1905.10437

# Roadmap 🗺️
The current roadmap is this:
 - [x] Collect initial datasets to use for the project
 - [ ] Write preprocessing steps that can do the necessary data transformations to get each dataset ready for modeling
 - [ ] Create some sort of scheme for the different datasets to note how to handle them for modeling
 - [x] Create starter notebooks with sample code for other people to visually explore the results and be able to grok the problem
 - [x] Write files that automate the modeling process and export results to a particular folder
 - [ ] Look at the results!  This is the most important part.  Look at the evals and parse what the results tell us.

# Current Questions ❔
Current issues that are being discussed are:

 - What rules should there be for pre-processing the data?
 - How much of the project can be done locally, vs. doing over the cloud?

# Installation 🖥️
To install the project on your computer locally, please follow these steps:

1).  Fork the repo to your own account

2).  Copy the URL of the git file (this is not the public url of the repo)

3).  From your command line or in Github Desktop run the command `git clone git_repo_location`

To install the dependencies, you can then run the commands:

### Regular Python 🐍
`cd time-series-ensembles`

`pip install -r requirements.txt`

### With Anaconda 🐍
If you'd like to create a development environment with anaconda, you can install everything you need with the `environment.yaml` file by following these steps:

First navigate into the root directory of the repo with this line:

`cd time-series-ensembles`

`conda env create -f environment.yaml`

If you then want to activate that environment you can do so with the command:

`conda activate time_series_ensembles`

### Creating Results

To recreate results for a particular experiment with a particular dataset, navigate to the root folder of this repo and run the following command:  `python -m src.experiment "experiment name" "dataset name"`

An example of this command would be: `python -m src.experiment "linear regression detrended demeaned" "car_parts"`

The current list of datasets and experiments that have been conducted are listed below:

| Dataset   | Experiment Name                      |
---------------------------------------------------
| car_parts | linear regression detrended demeaned |

Other arguments can be found in the `experiment.py` file to specify experiment parameters, if you wish.