# snek
Classifying snakes. 

## Challenge

AI Crowd challenge: https://www.aicrowd.com/challenges/snake-species-identification-challenge

Dataset: https://www.aicrowd.com/challenges/snake-species-identification-challenge/dataset_files

## Setting up the virtual environment

We recommend using python-virtualenv.
First, create a virtual environment and activate it.

```bash
$ python -m virtualenv ./.venv
$ source ./.venv/bin/activate
```

Now, install all the dependencies, using pip.

```bash
$ pip install -r requirements.txt
```

Now, you are set to go. Go around and mess with the scripts in the module `snek`.

## Directory Structure

```
snek/
|
|   .venv/
|   snek/
|   |   <python scripts...>
|   datasets/
|   |   train/
|   |   test/
|   |   cropped_train/
|   |   cropped_test/
|   models/
|   |   <trained models...>
```

## Scripts

We have made multiple scripts. Their descriptions are given below:

1. augment_data.py: Used the augment the dataset (Specify the input and output directories of the dataset as arguments)
2. basic_code.py: Used to run various models (resnet, vgg, densenet, inception_v3 and resnext50_32x4d). Run `python basic_code.py -h` for more details.
3. calc_mean_var.py: Used to calculate the mean and variance of the dataset provided. Run `python calc_mean_var.py -h` for more details.
4. clean_image.py: Resize and convert the image to grayscale.
5. cropping-images-using-trained-model.py: Python script to crop out the snake from an existing object recognition model.
6. gen_test_data.py: Generate test data from the train data by performing a 80-20 split randomly.
7. getpreds*.py: Get predictions of a provided model. Run `python getpreds*.py -h` for more details.
8. logistic_regression.py: Perform a classification by using logistic regression. Run `python logistic_regression.py -h` for more details.
9. plot_roc.py: Script to plot ROC Curve of given dataset/model.
10. remove_corrupted.py: Remove/clean the dataset of corrupted images.
11. train_efficientnet.py: Train an efficientnet model using the provided dataset.

## Authors

1. Aniket Pradhan
2. Bhavya Verma
3. Siddharth Nair
