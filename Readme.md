This is a Go AI
====

DISCLAIMER: This is a work in progress. Meaning that any file can not work/not be up to date for any reason. If you wish to see the clean version, please visit my repository CleanGoAI.

Using two convolutional neural networks and a tree search, this program is able to play go at a considerably advanced level.

Requirements: Numpy and Tensorflow need to be installed. Tested with tensorflow 1.2.0, numpy 1.13.0 and python 3.6.1. 

You can play go against the network by running `python3 play.py`.

If you wish to play against a more sophisticated version, which uses search trees and a second network, you first need to download network connections.

To do this, acquire pregenerated connections from [this Google Drive](https://drive.google.com/drive/folders/1uBrdv2Taka41yfc5b2lFRQPH30UjUOB0?usp=sharing) (download into the same directory as other files).

Then you can play go against the network by running `python3 play_long.py`.

You can also train the networks yourself by running `python3 policy_network_km.py` (necessary for all versions) and `python3 value_network.py` (only for play_long).
