Code for training / finetuning the Triplet Loss using a pretrained Mobilenet V2 and YooX dataset.

The main is the `main.py` file, parameters for the train are stored in the configuration file `config.json`.

The `neptuneLogger.py` file contains specific code for logging and train monitoring, using the online tool "neptuneAI".

Weights are cached in the `data` folder.

The Triplet Loss code is in the `utils.py` file.

`requirements.txt` provides all the required packages.

`ComposeDataset.py` contains the dataloader for feeding the dataset to the network.

The backbone and the overall model are in `models/GarnmentsNet.py`

The `logs` folder contains all the local log files that monitor the training procedure.

In order to run the training code is necessary to edit the `config.json` file with hyperparameters and run 'python3 main.py'.

The Dataset for the training procedure needs to have this relative path '../DatasetFolder/DatasetCV' from the current folder.

The original dataset needs to be modified in order to substitute the mannequin images with their respective cropped item.

The csv files for training should have this relative path : '../DatasetFolder/DatasetCV/{category}/train_pairs.txt'  from the current folder.

THIS CODE SHOULD BE RUNNED ON A LINUX SYSTEM