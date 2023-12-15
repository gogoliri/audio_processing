# audio_processing

Group project for the course COMP.SGN.120-2023-2024-1 Introduction to Audio Processing

## Installation

Instruction for installing dependencies goes here

install libraries

'pip install -r requirements.txt'

## Usage

generate normalized data

'python3 data.py'

This data.py file is just there to show how we generate normalized data

We already generated the normalized data and include them in the folder "normalized_*"

After install the libraries, open the notebook and run the whole notebook

Run the experiments in notebook "experiments.ipynb"

It just take 30s to run the whole notebook with full dataset

So we refer to submit the whole project

## File structure

- data folder: include the raw data and normalized data, notebook calls for those folders

analysis.py: the file contains functions to extract features such as energy, rms, zcr,

log spectrogram, logmel spectrogram, constant Q transform, and MFCCs

notebook calls for this file

- data.py: the file contains functions to normalized raw data and save to normalized_* folders

- experiments.ipynb: main notebook. display histograms, plots and train neural network

COMPSGN120_Audio_Project.pdf: the report file of the project

README.md: instruction file. 

requirements.txt: necessary python libraries

LICENSE.txt: auto generated prorietary license

## Authors

Khoa Pham Dinh (khoa.phamdinh@tuni.fi)
Uyen Phan (uyen.phan@tuni.fi)
