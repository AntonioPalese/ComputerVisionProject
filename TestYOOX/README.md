Code for computing a retrieval clothing query on 
YOOX dataset.

The `images` directory get the full input image.

The `outputs` directory is where the output (yoox items) for the input query are stored.

In `models` is stored the finetuned model of MobilenetV2.

In `parsed` should be stored the parsed version of the input image.

The `ExtractParsed.py` file extract the parsed clothes from the segmented image.

The `UtilsExtractParsed.py` file contains the utility code for generating the csv file for input and parsed.

The `palette.json` file is the palette list for extracting the segmentations.

In the `inputs` folder are automatically stored the inputs for the network after the extraction from the segmented file.

The only requirements for the user is to store in the `images` folder the input image to retrieve and in parsed, the respective same segmented image.

The `ComposeDataset.py` file contains the dataloader for the YOOX dataset.