#! /usr/bin/env python

# Standard library imports.
import subprocess
from pathlib import Path

# 3rd party imports. This is inefficient but common with fastai repo
from fastai2.vision.all import *
from fastai2_audio.core import *
from fastai2_audio.augment import *

# Majority of the code below is a snapshot from the DataChallenge notebook.
# It highlights the best case scenario achieved over several trials and
# for more details a README file provides info over the approach and challenges. 
def mergeSignal(x:AudioTensor) -> AudioTensor:
  "Merges the audio signals for inputs which have more than one signal, distortion is audibly minimum"
  return x.mean(dim=0).reshape(1,-1)


def audio_learner(dls, arch, loss_func, metrics):
  """Prepares a `Learner` for audio processing by changing the channel dimesion as per the input
  :param dls: Data Bunch object
  :param arch: Base architecture for training
  :param loss_func: Loss function to train the model
  :param metrics: Metrics to keep track and display during training
  
  """
  def _alter_learner(learn, channels=1):
    "A private method to augment a base learner for 1 channel"
    learn.model[0][0].in_channels=channels
    learn.model[0][0].weight = torch.nn.parameter.Parameter(learn.model[0][0].weight[:,1,:,:].unsqueeze(1))
  learn = Learner(dls, arch, loss_func, metrics=metrics)
  n_c = dls.one_batch()[0].shape[1]
  if n_c == 1: _alter_learner(learn)
  return learn


def train(data):
  "Trains a model using the best configuration and saves it"

  # Below transform converts audio signal to MFCC signal with provided mel-parameters
  aud2mfcc = AudioToMFCC(n_mfcc=64, melkwargs={'n_fft':1000, 'hop_length':200, 'n_mels':240})
  # Crops the signal to 2 seconds and pads the empty part with repetition
  crop = CropSignal(2000, pad_mode='repeat')
  # Overall transformation pipeline in-order. Delta applies a partial torchdelta per-channel.
  tfms = [mergeSignal, crop, aud2mfcc, Delta()]

  # DataBlock API, it takes the type of input and output i.e. AudioBlock and CategoryBlock, uses
  # a getter method `get_audio_files` and a setter method `get_y` with above transformation and
  # a random splitter with 80/20 split between train/val.
  ds = DataBlock(blocks=(AudioBlock, CategoryBlock),  
                 get_items=get_audio_files, 
                 splitter=RandomSplitter(),
                 item_tfms = tfms,
                 get_y=lambda x: str(x).split('-')[1][0])

  # Data bunch provides the dataloader in conjunction with DataBlock 
  dbunch = ds.dataloaders(data, bs=128)

  # Learner is a fastai class that defines a an object for training, validation and more.
  learn = audio_learner(dbunch, xresnet18(), CrossEntropyLossFlat(), accuracy)

  # `fit_one_cycle` uses the one-cycle training policy
  learn.fit_one_cycle(10, 1e-2)
  learn.unfreeze() # makes all layers trainable
  learn.fit_one_cycle(10, 1e-3)
  learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
  learn.save('bestConfigFullData')


if __name__ == '__main__':
  # pass the data path to you data folder. It can contain sub-folders with more files.
  data = Path('Data')
  train(data)