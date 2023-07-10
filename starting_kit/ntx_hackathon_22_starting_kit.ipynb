######################################################################################################
#  ________   _________   ___    ___ ___  ___  ________  ________  ___  __      _______   _______
# |\   ___  \|\___   ___\|\  \  /  /|\  \|\  \|\   __  \|\   ____\|\  \|\  \   /  ___  \ /  ___  \
# \ \  \\ \  \|___ \  \_|\ \  \/  / | \  \\\  \ \  \|\  \ \  \___|\ \  \/  /|_/__/|_/  //__/|_/  /|
#  \ \  \\ \  \   \ \  \  \ \    / / \ \   __  \ \   __  \ \  \    \ \   ___  \__|//  / /__|//  / /
#   \ \  \\ \  \   \ \  \  /     \/   \ \  \ \  \ \  \ \  \ \  \____\ \  \\ \  \  /  /_/__  /  /_/__
#    \ \__\\ \__\   \ \__\/  /\   \    \ \__\ \__\ \__\ \__\ \_______\ \__\\ \__\|\________\\________\
#     \|__| \|__|    \|__/__/ /\ __\    \|__|\|__|\|__|\|__|\|_______|\|__| \|__| \|_______|\|_______|
#                        |__|/ \|__|
######################################################################################################
#
# Data exploration and example file for submission in the NTX Hackathon challenge
#
######################################################################################################

import mne
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# download data https://filesender.renater.fr/?s=download&token=e1de0ec4-09bc-4194-b85b-59830cb04af3
# download test data from https://codalab.lisn.upsaclay.fr/competitions/8336

# Path to training data
train_path = "path/to/NTX-hackathon22/training/"
# Path to testing data (public test set)
test_path = "path/to/NTX-hackathon22/testing/"
condition = "EC"  # use only closed eyes condition for demonstration purpose
train_subj = 10  # use 10 instead of 1200 training subjects, for demonstration purpose
test_subj = 400  # use 10 instead of 400 testing subjects, for demonstration purpose

train_raws = []
for s in range(1, train_subj + 1):
    fname = f"subj{s:04}_{condition}_raw.fif.gz"
    raw = mne.io.read_raw(train_path + fname, preload=True)
    train_raws.append(raw)
test_raws = []
for s in range(1201, 1201 + test_subj):
    fname = f"subj{s:04}_{condition}_raw.fif.gz"
    raw = mne.io.read_raw(test_path + fname, preload=True)
    test_raws.append(raw)

# Visualisation of the sensor position in 3D
train_raws[0].plot_sensors(show_names=True, kind="3d")
# Filtering the signal and plotting all 128 channels
train_raws[0].notch_filter([60.0, 120.0, 180.0]).filter(l_freq=1, h_freq=30)
train_raws[0].plot(duration=5, n_channels=129, color={"eeg": "darkblue"})
# Marking some channels as bad (flat electrodes, disconnection, noisy signal, ...)
train_raws[0].info["bads"] = [
    "E7",
    "E48",
    "E106",
    "E112",
    "E55",
    "E31",
    "E105",
    "Cz",
    "E80",
    "E81",
    "E88",
]
# Plotting power spectrum for all electrodes but the bads.
train_raws[0].plot_psd(fmax=50)

# Get ndarray from MNE raw files to generate train and test input
X_train, X_test = [], []
crop_start, crop_end = 5, 15  # use only a 10s window, from 5s to 15s
for r in train_raws:
    X_train.append(r.copy().crop(tmin=crop_start, tmax=crop_end).get_data())
for r in test_raws:
    X_test.append(r.copy().crop(tmin=crop_start, tmax=crop_end).get_data())

# get the age to predict from the CSV file
meta = pd.read_csv(train_path + "train_subjects.csv")
y_train = []
for age in meta["age"][:train_subj]:
    y_train.append(age)

# Create sklearn pipeline, fit and predict
ppl = make_pipeline(Vectorizer(), PCA(n_components=6), Ridge(alpha=0.5))
ppl.fit(X_train, y_train)
y_pred = ppl.predict(X_test)

# create submission file
dummy_submission = []
for subj, pred in zip(range(1201, 1201 + test_subj), y_pred):
    dummy_submission.append({"id": subj, "age": pred})
pd.DataFrame(dummy_submission).to_csv("mysubmission.csv", index=False)

# zip the csv file (without anything else) and submit it on the website!
