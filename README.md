
# Emotion Detection Model

This repository contains the code and resources for an emotion detection model using Convolutional Neural Networks (CNNs). The model is trained on grayscale images to classify emotions into seven categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Emotion Detection Model leverages deep learning to identify facial expressions in grayscale images. It uses TensorFlow and Keras libraries for implementation and training, with a focus on efficient data preprocessing and model optimization.

## Directory Structure

```
project/
├── model/
│   ├── emotion_model.json    # Model architecture
│   ├── emotion_model.h5      # Trained weights
├── data/
│   ├── train/                # Training dataset
│   ├── test/                 # Testing dataset
├── TrainEmotionDetector.py   # Script to train the model
├── EvaluateEmotionDetector.py# Script to evaluate the model
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/emotion-detection.git
   cd emotion-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `data` directory contains your training and testing datasets structured as required.

## Usage

### Training the Model

To train the model, run the `TrainEmotionDetector.py` script:
```bash
python TrainEmotionDetector.py
```
This script will:
- Load the training data from the `data/train` directory.
- Train the CNN model for 50 epochs.
- Save the trained model as `emotion_model.json` and its weights as `emotion_model.h5` in the `model/` directory.

### Evaluating the Model

To evaluate the trained model, run the `EvaluateEmotionDetector.py` script:
```bash
python EvaluateEmotionDetector.py
```
This script will:
- Load the trained model and weights.
- Evaluate its performance on the test dataset in `data/test`.
- Display a confusion matrix and classification report for further analysis.

## Model Details

The model architecture includes:
- **Convolutional Layers**: Extract spatial features from input images.
- **Pooling Layers**: Downsample feature maps.
- **Dropout Layers**: Prevent overfitting.
- **Fully Connected Layers**: Perform final classification.

The input shape is `(48, 48, 1)` (grayscale images of size 48x48 pixels).

### Emotion Classes
- **0**: Angry
- **1**: Disgusted
- **2**: Fearful
- **3**: Happy
- **4**: Neutral
- **5**: Sad
- **6**: Surprised

## Requirements

This project requires Python 3.6 or later. Key dependencies include:
- TensorFlow 2.4.1
- Keras 2.4.3
- NumPy 1.19.5
- OpenCV 4.5.1
- Scikit-learn

For the full list of dependencies, see [requirements.txt](requirements.txt).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
