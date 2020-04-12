# Audio deep learning challenge

### Note: Inference will not work because the uploaded model is a dummy file. Please clone and run full_train.py with your data.

## Overview
- Task: To create a 'good' classification model on audio data
- Target:
    - Clean Code
    - Comments/ Doc-strings
    - What; Why; Results
    - Caveats, assumption and future possibilities

- Deliverables:
    - Notebook
    - Docker container
    
## Instructions:
Once you clone, you need to run the below commands in order:

`$ docker build -t audio_deep_learning:latest .`

`$ docker run -v path/to/host/folder:/app/data --rm -it audio_deep_learning`


Above build commands take about 4 mins, depending on your network speed. When you run the container, it prints the usage for the inference API with examples and starts bash for you. The second command also allows you to mount a local folder from your machine to test audio files.

Test calls will look like below commands, I'm also attaching a snapshot of the output from below commands to show what is expected.

`$ inference example.wav`

`$ inference path/to/test_folder`
