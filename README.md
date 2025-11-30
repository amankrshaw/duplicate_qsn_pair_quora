# README

## Project Overview

This project explores various text preprocessing strategies used in Natural Language Processing (NLP) and builds an Artificial Neural Network (ANN) classifier model to detect duplicate question pairs. The process includes data preprocessing, feature engineering, model building, training, and evaluation.

## Introduction to Text Preprocessing in NLP

## Data Exploration and Visualization
- **Loading the Dataset**: Importing the dataset of question pairs.
- **Initial Exploration**: Examining the dataset for structure, missing values, and basic statistics.
- **Visualization**: Creating visual representations to understand the distribution.

## Text Preprocessing Implementation
- **Cleaning and Normalization**: Applying tokenization, lowercasing, HTML tags removal, punctuation removal, stop words removal, and stemming/lemmatization to preprocess the text.
- **Vectorization**: Transforming the cleaned text into numerical vectors suitable for model input using Word2Vec(gensim).

## Feature Engineering
- **Creating Various Features**: Created several features(eg. length of sentence, first and last word same or not, word share, common words, etc.) including fuzzy features to enhance the model's ability to detect similarities.

## Building the ANN Classifier
- **Model Architecture**: Designing the neural network structure, including input layers, hidden layers, activation functions, and output layers.
- **Compilation**: Setting the optimizer, loss function, and evaluation metrics.
- **Training**: Training the model on the preprocessed dataset, with considerations for batch size, epochs, and validation split.
- **Evaluation**: Assessing the model's performance using metrics such as accuracy, precision, recall, and F1 score.
- **Hyperparameter Tuning**: Adjusting the model parameters to improve performance.

## Model Training and Evaluation
- The model was trained on 100,000 rows of data for 100 epochs.
- **Training Accuracy**: 0.87
- **Validation Accuracy**: 0.80
- **ROC AUC Score**: 0.87
- **F1 Score**: 0.72

## Conclusion and Insights
- **Model Performance**: The ANN classifier showed good performance with a training accuracy of 0.87 and a validation accuracy of 0.80. The ROC AUC score of 0.87 and an F1 score of 0.72 indicate a robust model capable of detecting duplicate question pairs effectively.
- **Challenges and Future Work**: Further improvements can be made by experimenting with different neural network architectures, feature engineering techniques, and hyperparameter tuning.
