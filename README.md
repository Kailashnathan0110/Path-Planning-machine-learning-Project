 # Neural Astar for Path planning 
 
This is the repository for MAE 551 Applied machine learning for mechanical engineers final project.

## Project Contributers
Barath Sundaravadivelan ([bsunda10@asu.edu](mailto:bsunda10@asu.edu)) <br />
Kailash Nathan Ramahlingem ([kramahli@asu.edu](mailto:kramahli@asu.edu))<br />


## Initial Setup 
Please enter the scripts in your terminal to install the conda environment and the necessary prerequisites. Tested in python version 3.9.

```shell
$ git clone https://github.com/Kailashnathan0110/Path-Planning-machine-learning-Project.git
$ git checkout  origin/branch-Master-v1-MAE551-PathPlanning-OutputGenerator
$ conda create --name projVenv python==3.9
$ conda activate projVenv
(projVenv) $ pip install -r requirements.txt
```
## Selecting the Dataset 
Please go to Configurations -> trainingConfig.yaml
```shell
# Select any one of the training Data (Value) and paste your selection as string at trainingData (Key)
# 1. multiple_bugtraps_032_moore_c8
# 2. mixed_064_moore_c16
# 3. all_064_moore_c16

trainingData: "multiple_bugtraps_032_moore_c8"
```
You can change training dataset with the desired ones.


## Training the model
```shell
(projVenv) $ python modeltrain.py
```

## Generating output
```shell
(projVenv) $ python output.py
```
The Gifs will be generated in the directory "OutputResult"

# Sample
Below are the sample gifs for reference <br />

## NeuralAstar generated Result
![](OutputResult/Video_mixed_064_moore_c16_nastar.gif)

## Astar generated Result
![](OutputResult/Video_mixed_064_moore_c16_astar.gif)




