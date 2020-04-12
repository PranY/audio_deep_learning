# Audio deep learning challenge

## Note: Inference will not work because the model uploaded is a dummy file. Please clone and run full_train.py with your data.

## Overview
- Task: To create a 'good' classification model on audio data
- Ask:
    - Clean Code
    - Comments/ Doc-strings
    - What; Why; Results
    - Caveats, assumption and future possibilities

- Deliverables:
    - Notebook
    - Docker container


### Approach
- MVP
    - Collect the data and research about pre-processing steps for audio data.
    - Create a classification model and get baseline accuracy.
    - Test an end-to-end pipeline.

- Iteration
    - Add relevant transformations and check compatibility with data block API.
    - Since we are working with images, focal loss may be a good choice in this situation. The idea is that the signal-to-noise ratio is low in these images and focal loss helps it. Although pytroch doesn't support focal loss, will use any public implementation.
    - Create an API to support prediction for any .wav file or a folder with .wav files and integrate with docker. Mount points need to be defined to access local files on host machine.

---

#### EDA
- A quick scan of data1.zip shows that the audio-file length is between 0 and 4 seconds. The max length may be longer in the overall data.
- Test instructions clearly highlight that there are 10 classes.
- The medium article on [audio classification from Dessa](https://medium.com/dessa-news/detecting-audio-deepfakes-f2edfd8e2b35) explains the solution.
- Wav data -> Spectrogram -> A simple vision model -> Class prediction
- The article also mentions masked-pooling to avoid over-fitting.

#### Data preparation
- Because the audio files are of different lengths, I'll use either a zero-padding or replicate the same sound [~~ToDo: Figure out how to do this~~]
- [This location](https://musicinformationretrieval.com/index.html) explains a great deal about audio-preprocessing, reading the bare minimum to get the MVP
- Converting to Mel-Spectrogram seems to be the general approach. Will follow with MFCC and MFCC + Delta if time permits.
- It is also common to remove silence from audio, I think that's a bad idea because it breaks the temporal continuity of sound but I'll still try.

#### Findings/Log
- Reading the data in librosa and analyzing, the audio file and sampling rate comes from load function. There are many transforms but they won't bind easily to pytorch.
- There is an experimental extension for fastai for audio. It follows fastai's transforms-based API and all I need to figure out is how the data can work with the DataBlock API.
- The data is read easily with the `AudioTensor` class and some files have 2 audio-wave signal and some have one. I'll average the information over the first dimension and listen if there is a severe degradation in sound quality, if not I'll use it.
- The averaging part works but the spectrograms are of different dimensions, experimenting with the parameters of MelSpectrogram class.
- The parameters change the information of the spectrograms but the dimensions remain the same until I change the mel-Spectrogram channels. Changing the cropping point of the signal adds more information but dimensions are still not matching and cannot be loaded into a batch.
- After reviewing the `AudioTensor` and `CropSignal` source code I can confirm that both the class and transform works fine, the issue is with the sampling rate. Coincidentally the samples I used in EDA had the same sampling rate information but it varies with file.
- Reviewing `librosa.load()` to check if I can use it with transforms.
- librosa hack doesn't work with `AudioTensor` in this transforms pipeline, the loading is very slow.
- GOTCHA, fixing the sampling rate at 16K for all files outside the pipeline using bash script. [Todo]: Write a multi-threaded version later. Adds a new dependency `sox` but it's lightweight and works quickly. [Update: Wrote a python-ish version to use with the inference API]
- Now the tensor shapes are consistent to create a data bunch using fastai DataBlock API
- Testing baseline with 10% data and default MelSpectrogram config taken from [Dessa Repo](https://github.com/dessa-oss/fake-voice-detection/blob/master/code/constants.py)
- [ToDo] Most params for MelSpectrogram are powers of 2 in the examples provided by the library, for example n_mels=128, n_fft=1024, hop_length=128. This is helpful for GPU processing as Nvidia's support is optimized for this. Need to re-test with these numbers after a strong baseline. [Update: these numbers perform similar but Dessa's defaults are better]
- Learner doesn't work, the stem-model xresnet has channel issues. Kevin B from fastai provided a hack on forum to alter the learner for single channel, using that.
- Learner is changed to single-channel, will write an adaptable extension later to support any number of channel.
- <u>**Training works as expected, first baseline gets 80% accuracy using 10% data.**</u>
- This library supports the [recent masking research](https://arxiv.org/abs/1904.08779) for both frequency and time channel. Awesome, let's test that with the transforms API.
- Masks are helping, experimenting with few random configurations.
- Both frequency and time mask at size 5 work better, finalizing for baseline update
- <u>**Baseline updated with ~85% accuracy on 10% data, medium accuracy reached**</u>
- The predict method doesn't work for learner, reading source code.
- GOTCHA, learner requires a flattened loss function as provided with fastai library, this binds back to callbacks and transforms API. Predict method works now and it allows the use of interp object. I need to custom write a flattened version for Focal Loss incase I get the time to implement it. [Update: Implemented the flattened version and training works fine but few callbacks fail during prediction because `AudioTensor` doesn't support some attributes required for callbacks. Will raise this to fastai2_audio later and send a PR]
- MFCC provides Discrete Cosine Transforms over transformed mel-signal. The MFCC decomposition is proven better for getting spectrograms. The library also support `Delta` augmentation which applies a partial torchdelta of different orders on each channel.
- Cropping signal for 2 secs works best and models trains better without `RemoveSilence`. This can be due to the loss of temporal continuity as expected earlier but still a bit unclear.
- <u>**Model is now at 99% accuracy on 10% data**</u>. I have pushed the model to over-fit on this data to quickly test on a new folder as test case. Model performs ~60% on the `data2` folder which is expected because the class distribution caused our model to over-fit on that distribution and class-0 and class-2 are confused often in this new sample.
- Fine tuning the model on complete data with same 80/20 random split.
- <u>**Model trained on full-data achieves 95% accuracy measured on a 20% random validation set**.</u>

---

### Future work

- The accuracy is reasonable but it can be improved further with more domain knowledge around Spectrograms and other audio-processing techniques. 48 hours limited the amount of information I could gather and absorb as this was the first time I'm working with audio data. *Takeaway: Use domain knowledge to improve the data augmentation.*
- I wanted to test Focal Loss but found few limitations in the library. Given more time I can fix those issues and update the library and re-test. *Hunch: Trying this loss with better augmentations can add another ~2-3% in accuracy.*
- Because we are dealing with images on the classification side, I can use progressive sizing of images to improve the model. Test-Time augmentation and mixup augmentation can also help though I may need to replicate single channel images to 3 channel for that to work.
- Normalization of image data plays a crucial role in training and because I have no domain knowledge, I did not use any normalization. This must have lead to distribution shifts after ReLU activation in deeper layers and it reduces the overall efficiency of the model by making more zero tensors. *Takeaway: Find out more on spectrogram normalization.*
- The model could be improved further with more data and I can use larger batch sizes with mixed-precision training supported with fastai. [Larger batch sizes will help](https://arxiv.org/abs/1711.00489) generalize better and the model may converge sooner. *Overall, I'm guessing we can achieve ~99% accuracy in this task, which on real-world data will still be great if not 99%.*
