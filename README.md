# Bayesian Linear Bandits for Large-Scale Recommender Systems

This repository contains the source code for the paper "Bayesian Linear Bandits for Large-Scale Recommender Systems".


## Getting Started 

#### Clone this repository
```
git clone https://github.com/guchis/reduction_bandits
cd reduction_bandits 
```

#### Setup the environment

The [environment.yml](https://github.com/guchis/reduction_bandits/blob/main/environment.yml)
file describes the list of library dependencies. For a fast and easy way to setup the environment we encourage you to
use [Conda](https://docs.conda.io/en/latest/miniconda.html). Use the following commands to install the dependencies in
a new conda environment and activate it:
```
conda env create -f environment.yml
conda activate reduction_bandits
```

## Running experiments

### Amazon Review Data 
Download a subset of amazon review data (all subsets available [here](https://nijianmo.github.io/amazon/#subsets)):
```
curl "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games.csv" --create-dirs -o "dataset/amazon/Video_Games.csv"
```
Preprocess the data (create context vectors and rewards):
```
python 5_preprocess_amazon.py
```
Run the evaluation for 1000 timesteps with reduction matrix of dimension 30 and averaging results of every policy over 3 runs:
```
python 3_run_evaluation.py -t 1000 -d 30 --num-rep 3 --data amazon
```
Description of parameters of the script:
```
usage: 3_run_evaluation.py [-h] -t TRIALS -d DIMENSION [--num-rep NUM_REP] [--load-old-reduct-matrix] [--data DATASET_TYPE]

options:
  -h, --help            show this help message and exit
  -t TRIALS, --trials TRIALS
                        Number of trials
  -d DIMENSION, --dimension DIMENSION
                        Dimension of reduction matrix
  --num-rep NUM_REP     Number of repetitions for each algorithms, default = 5
  --load-old-reduct-matrix
                        Load reduction matrix of needed dimension from file. Will throw error if no file exists.
  --data DATASET_TYPE   Which data to use, 'amazon', 'movielens' or 'jester'
```
Evaluation script uses [evaluation.json](https://github.com/guchis/reduction_bandits/blob/main/config/evaluation.json)
configuration file. Edit it to set the hyperparameters and choose which
policies to use in the experiment.

To tune the hyperparameters of a policy create or edit a corresponding [json config file](https://github.com/guchis/reduction_bandits/blob/main/config/bcmabrp.json)
and run:
```
python 4_tune_parameters.py -t 1000 -d 30 --num-rep 3 --data amazon --config config/bcmabrp.json
```

## MovieLens

Download and unzip the MovieLens 10M dataset:
```
curl "https://files.grouplens.org/datasets/movielens/ml-10m.zip" --create-dirs -o "dataset/movielens/ml-10m.zip"
unzip -j dataset/movielens/ml-10m.zip -d dataset/movielens
```
**Note**: You can use other versions of the MovieLens dataset as well, as long as they contain movies.dat and ratings.dat files.

Preprocess the data (create user and action contexts and reward vectors):
```
python 1_preprocess_movielens.py
```

Run the evaluation for 1000 timesteps with reduction matrix of dimension 30 and averaging results of every policy over 3 runs:
```
python 3_run_evaluation.py -t 1000 -d 30 --num-rep 3 --data movielens
```

To tune the hyperparameters of a policy create or edit a corresponding [json config file](https://github.com/guchis/reduction_bandits/blob/main/config/bcmabrp.json)
and run:
```
python 4_tune_parameters.py -t 1000 -d 30 --num-rep 3 --data movielens --config config/bcmabrp.json
```

## Jester
Jester dataset comes prepackaged inside the [scikit-surprise](https://surpriselib.com/) package that we have already installed
in the [Getting Started](https://github.com/guchis/reduction_bandits#setup-the-environment) part. So to download the data and create all needed files just run:
```
python 2_preprocess_jester.py 
```

Then you can run the evaluation for 1000 timesteps with reduction matrix of dimension 30 and averaging results of every policy over 3 runs:
```
python 3_run_evaluation.py -t 1000 -d 30 --num-rep 3 --data jester
```

To tune the hyperparameters of a policy create or edit a corresponding [json config file](https://github.com/guchis/reduction_bandits/blob/main/config/bcmabrp.json)
and run:
```
python 4_tune_parameters.py -t 1000 -d 30 --num-rep 3 --data jester --config config/bcmabrp.json
```