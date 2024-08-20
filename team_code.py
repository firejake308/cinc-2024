#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import time
import torch
import torchvision

from helper_code import *
from cwip import CWIPModel

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    # records = records[:128] # TODO use the full dataset
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can remove this part of the code.

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    digitization_features = list()
    classification_features = list()
    classification_labels = list()

    # initialize the model
    model = CWIPModel().float()
    # model.cuda()
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((320, 640)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Iterate over the records.
    BATCH_SIZE = 16
    start_time = time.time() 
    for i in range(0, num_records, BATCH_SIZE):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        if time.time() - start_time > 60 * 5:
            print('Reached time limit')
            break
        # build a batch
        img_batch = []
        wav_batch = []
        for b in range(BATCH_SIZE):
            record = os.path.join(data_folder, records[i+b])
            img = np.array(load_images(record)[0])
            img = transf(img[:,:,:3]).float()
            wav, _fields = load_signals(record)
            wav = torch.tensor(wav.T).float()
            img_batch.append(img)
            wav_batch.append(wav)
        img_batch = torch.stack(img_batch, dim=0)
        wav_batch = torch.stack(wav_batch, dim=0)
    
        # do one step of the model training
        out = model(img_batch, wav_batch)
        lab = torch.eye(BATCH_SIZE)#.cuda()
        loss = loss_fn(out, lab)
        print(loss.item())
        loss.backward()
        optim.step()
    
    for i in range(num_records):
        record = os.path.join(data_folder, records[i])
        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        features = extract_features(record, model)
        
        digitization_features.append(features)

        # Some images may not be labeled...
        labels = load_labels(record)
        if any(label for label in labels):
            classification_features.append(features)
            classification_labels.append(labels)

    # ... but we expect some images to be labeled for classification.
    if not classification_labels:
        raise Exception('There are no labels for the data.')

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    digitization_model = np.mean(features)

    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.
    
    # This very simple model trains a random forest model with these very simple features.
    classification_features = np.vstack(classification_features)
    classes = sorted(set.union(*map(set, classification_labels)))
    classification_labels = compute_one_hot_encoding(classification_labels, classes)

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 6  # Number of trees in the forest.
    max_leaf_nodes = 17  # Maximum number of leaf nodes in each tree.
    random_state   = 42  # Random state; set for reproducibility.

    # Fit the model.
    classification_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(classification_features, classification_labels)

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, classification_model, classes, cwip_model=model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    digitization_model = joblib.load(digitization_filename)

    classification_filename = os.path.join(model_folder, 'classification_model.sav')
    classification_model = joblib.load(classification_filename)

    cwip_model = CWIPModel()
    cwip_model.load_state_dict(torch.load(os.path.join(model_folder, 'cwip.pt')))
    classification_model['cwip'] = cwip_model

    return None, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.

    signal = None

    # Run the classification model; if you did not train this model, then you can set labels = None.

    # Load the classification model and classes.
    model = classification_model['model']
    cwip_model = classification_model['cwip']
    classes = classification_model['classes']

    # Get the model probabilities.
    features = extract_features(record, cwip_model)
    probabilities = model.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose the class or classes with the highest probability as the label or labels.
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return signal, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record, model):
    images = load_images(record)
    mean = 0.0
    std = 0.0
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((320, 640)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # build a batch
    img_batch = []
    for image in images:
        img = np.array(image)
        img = transf(img[:,:,:3]).float()
        img_batch.append(img)
    img_batch = torch.stack(img_batch, dim=0)

    # do one step of the model training
    model.eval()
    out = model.img_block(img_batch)

    return out.cpu().detach().numpy()

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None, classes=None, cwip_model=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        d = {'model': classification_model, 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)

    if cwip_model is not None:
        filename = os.path.join(model_folder, 'cwip.pt')
        torch.save(cwip_model.state_dict(), filename)

