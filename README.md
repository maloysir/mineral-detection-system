
A machine learning based system for identifying and classifying mineral samples from RGB images.
The system uses a deep learning encoder and shallow classifier to analyze mineral textures and features and predict the mineral speicemen.

2. Overview

This project aims to automate mineral identification using computer vision and machine learning.

Key features:

Image based mineral classification

Deep learning feature extraction

Fast prediction through a trained model

Simple interface for uploading mineral images

The system loads a pre trained model and predicts the mineral class from the input image.

3. Model Architecture

The system uses a two stage deep learning pipeline:

Feature Encoder

Model: dino_encoder.pt

Extracts visual features from mineral images using a transformer-based encoder.

Classifier

Model: mlp_classifier.pt

A multi-layer perceptron (MLP) that classifies extracted features into mineral categories.

Pipeline:

Input Image
     ↓
Image Preprocessing
     ↓
DINO Encoder (feature extraction)
     ↓
MLP Classifier
     ↓
Mineral Prediction


Main model files:

model/dino_encoder.pt
model/mlp_classifier.pt
best_model.pth

4. Dataset

The model was trained using a dataset of mineral sample images.

Example minerals included:

Biotite

Gypsum

Epidote

Perovskite

Chondrite

Allogeneic Breccia

Meteorite (Seimchan)

Dataset structure example:

samples/
    Biotite.png
    gypsum.png
    epidote.png


Images represent different mineral textures and crystal structures.

5. Installation

Clone the repository:

git clone https://github.com/maloysir/mineral-detection-system.git
cd mineral-detection-system


Install dependencies:

pip install -r requirements.txt

6. Usage

Run the application:

python app.py


Steps:

Start the application

Upload a mineral image

The system processes the image

The predicted mineral class is displayed

7. Authors

Project contributors:

Chomangli

Mr. Maloy Dey

Dr.Dushmanta Kumar Das

Affiliation:

National Institute of Technology Nagaland (NITN)




classify_app/
 ├── app.py
 ├── model/
 │   ├── dino_encoder.pt
 │   └── mlp_classifier.pt
 ├── samples/
 ├── authors/
 ├── requirements.txt

