# Overview
This repository contains the code for the paper    "Physics-inspired adaptions to low-parameter neural network weather forecasts systems."
by Sebastian Scher and Gabriele Messori, under review at AIES.

# Scripts
The most important scrips in this repository:

`custom_layers.py` contains the tensorflow code for all custom convolution operations, implemented as tf.keras layers that can serve as drop-in replacements for normal 2d convolution layers.

`era5_to_tfrecord.py` converts the WeatherBench data to tensorflow records

`store_nc_shapes.py` stores the shape of the input data in a file, needed later on.

`era5_compute_normalization.py` computes the normalization weights on the training data

`era5_train_batch_tetralith.py` runs the neural network training for one configuration. The configuration is passed in as input parameters this script is written for the specific tetralith supercomputer, and might need to be adapted for other environments


`slurm_submission_scripts` contains shell scripts that submit the trianing, prediction and evaluation scripts. They are written for the specific tetralith supercomputer, and might need to be adapted for other environments

`era5_predict_batch.py` uses a trained model with a specific configuration to make prediction son the test set. The configuration is passed in as input parameters

`eval_net_fcs_batch.py`  evlautes the predictions produced by `era5_predict_batch.py`




# requirements
the requirements are conatined in the conda environment file `nn-reanalysis-env.yml`



