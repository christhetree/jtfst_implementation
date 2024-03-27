# Baseline framework

This is an implementation of the baseline Hidden Markov Model-based classification of playing evolution techniques (specifically, glissando and portamento). The following pipeline contains a separate set of installation guidelines and package requirement.

## Installation Guide

In addition to the python package requirements, the pipeline requires `vamp-plugins` and `sonic-annotator` for the pitch extractor. The installation is to be performed from the `baseline/` directory.\
**Note:** It is advised to create a separate python `virtualenv` for running the pipeline as it may have requirement conflicts with the JTFST implementation.   
```
pip install -r requirements_baseline.txt
sh setup.sh
```
Successful installation of the pitch extraction plugins will be indicated by `Pitch extraction plugin test successful`. 

## Training and Evaluation

The entire pipeline can by executed by the `run_pipeline.py` file. The pre-processing setting is hard-coded for the **Chinese Bamboo Flute Dataset** which needs to be downloaded before executing the pipeline. Evaluation is performed with 5-fold cross validation. This may, however, be quite time-consuming.

```
# Running without cross validation
python3 run_pipeline.py --data_dir=/path/to/dataset/

# Running with 5-fold cross validation
python3 run_pipeline.py --data_dir=/path/to/dataset/ --cvd=5
```
