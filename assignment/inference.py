#! /usr/bin/env python

# Standard library iports.
import subprocess
from pathlib import Path

# 3rd party imports. This is inefficient but common with fastai repo
import click
from fastai2.vision.all import *
from fastai2_audio.core import *
from fastai2_audio.augment import *

# Local imports.
from assignment.full_train import mergeSignal, audio_learner


def convert_file(infile: Path, bitrate: int = 16000) -> None:
    """Convert one file to the bitrate provided in the arguments.
    This file does not return anything but raises exception if sox file fails.
    
    :param infile: Input file path to convert.
    :param bitrate: The desired bitrate.
    :raises: OSError, SubprocessError
    """
    infile = Path(infile)
    tempfile = infile.parent / '.temp.wav'
    # Call sox to convert the file.
    subprocess.run(['sox', str(infile), str(tempfile), 'rate', str(bitrate)], 
                    check=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    
    # Remove the tempfile and replace the infile with the modified bitrate.
    subprocess.run(['mv', str(tempfile), str(infile)], 
                    check=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


def convert_all(input_dir: str, bitrate: int = 16000) -> None:
    """Converts all .wav files directly inside the folder to the desired bitrate.
    
    :param input_dir: Input directory with wav files to convert.
    :param bitrate: The desired bitrate.
    """
    for infile in Path(input_dir).glob('**/*.wav'):
        convert_file(infile, bitrate)


def load_audio_model(model_name: str):
    """Loads the model but creating a dummy data bunch and learner type. It needs to be
    consistent with the model type passed in the model name. The method loads the model
    with weights and optimizer state.
    
    :param model_name: Name of the model you wish to load
    """

    # Below pipeline is a trimmed-copy of full-training, please refer to full_train.py
    # for detailed explanation for each step.
    aud2mfcc = AudioToMFCC(n_mfcc=64, melkwargs={'n_fft':1000, 'hop_length':200, 'n_mels':240})
    crop = CropSignal(2000, pad_mode='repeat')
    tfms = [mergeSignal, crop, aud2mfcc, Delta()]

    ds = DataBlock(blocks=(AudioBlock, CategoryBlock),  
                    get_items=get_audio_files, 
                    item_tfms = tfms)

    dbunch = ds.dataloaders(Path('data'), bs=1)

    learn = audio_learner(dbunch, xresnet18(), CrossEntropyLossFlat(), accuracy)
    learn.load(model_name)
    
    return learn

@click.command()
@click.argument('sample')
def main(sample):
    """Inference is defined to predict the class type of input .wav file(s). User can provide
    either a path to .wav file or a folder consisting .wav files. The output is a tuple of
    provided file(s) and associated class.
    
    
    Example:
      \b
      $ inference example.wav
      $ inference path/to/test_folder
    """
    sample = Path(sample).resolve()
    model = load_audio_model('bestConfigFullData')
    
    if sample.is_dir():
        # Got a directory, convert all files.
        convert_all(sample)
        samples = list(sample.iterdir())
        for sa in samples:
            try:
                cls = model.predict(sa)[1]
                print(sa, cls)
            except Exception as ex:
                print(f'Failed to predict {sa} due to {ex}')
    else:
        # Got a single file, so run only one file.
        convert_file(sample)
        print(str(sample), model.predict(sample)[1])
