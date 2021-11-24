# Online Multichannel Source Separation Random-Directions Demo

To let the notebook run in Google Colab, click on the colab button, and then, for instance, Runtime-Run all.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TUIlmenauAMS/LowDelayMultichannelSourceSeparation_Random-Directions_Demo/blob/main/online_multichannel_source_separation_random_directions_demo.ipynb)

This is an online multichannel audio source separation demo, including the random directions software for own experiments, as described for instance in the Asilomar Conference on Signals, Systems, and Computers 2021 talk and paper. A quicker file based demo can be found in:

https://github.com/TUIlmenauAMS/LowDelayMultichannelSourceSeparation

But the present notebook also contains the separation algorithms, such you can run them by running the notebook, and listen to the result. Also the audio source files for the room simulation can be replaced by other files, or the output of the room simulation can be replaced by other multichannel audio files, which already contain mixed audio sources. You can change the audio sources for the room simulation in the "Simulated Room" cell in line 7, "files=", or by providing a multichannel mixed file with a different name there in line 6, "mixfile=". If you upload a file into Colab, using the folder and upload icon on the left, use the path "/content/..." for your file name in the program (otherwise there will be an error message).

Gerald Schuller, November 2021.
