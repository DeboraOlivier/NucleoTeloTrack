## How to install jupyter notebooks

### Dependent packages

- conda create -n nucleotelotrack python=3.6
- conda activate nucleotelotrack
- pip install pandas seaborn scikit-image networkx pathos tqdm notebook jupyter_contrib_nbextensions

## How to use jupyter notebooks

- set the different input/ouptut parameters
- run the notebook:
  - if you want to run cell by cell use shift + Enter
  - or you can just run the entire notbook

## Process_Data notebook
This notebook will...

select tracks
exclude objects at teh bordres
exculde violated state transition according to rule graph
create a csv per position with all tracks and the selected features

## Analyze_Data notebook
This notebook will ...

## Todo

trackprocessor.python
- process_data(): add option for color map, time step in minutes.

## Issues
