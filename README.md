# rsna
RSNA Intracranial Hemorrhage Detection - https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview

We used PyTorch and image augmentation to train a CNN to detect hemorrhages from images of brains.
The approach is to use **transfer learning**, that is using a pretrained CNN with its weights and only optimizing the final layer to adapt the network to our needs.

The repo contains two notebooks:
- eda: a short exploratory data analysis to get familiar with the data. 
We show that the train dataset contains a lot of "empty" images (ie. containing few bright pixels).
They are removed from the training of the CNN.

- modelling: training of the network, creation of checkpoints to save progress and prediction on the test set.

The python files:
- calculate_image_metadata: generates two csv files mapping every training image id with:
  1. the sum of the pixel values of the image
  2. the number of black pixels (pixels equal to 0)
- create_test_file: reshapes the sample submission csv from kaggle to something suitable for predictions
- filter_images: generates a csv file containing only the non-empty image ids, for the training phase
- util: contains several functions to load and save a PyTorch model, create the submission file and split the training data between train and validation files
- xception: source code of the Xception model as defined in https://arxiv.org/pdf/1610.02357.pdf
