# Student Dropout & Academic Success Prediction (BPNN from Scratch)

A **Backpropagation Neural Network (BPNN)** implemented from scratch using NumPy to classify university students into three outcomes: **Graduate**, **Dropout** or **Enrolled**.

## Overview
Early identification of students at risk of dropping out helps institutions provide timely support. This project builds a multi-layer neural network without deep learning frameworks. Forward propagation, backpropagation and gradient descent are all written manually.

## Dataset
[Predict Students' Dropout and Academic Success (UCI)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success): 4,424 student records with demographic, socio-economic and academic features.

## Methodology
1. **Data cleaning:** removed duplicates and missing values, encoded target labels
2. **Feature selection:** dropped 5 less relevant columns (31 features remaining)
3. **Outlier handling:** winsorisation (clipped to 5th–95th percentile)
4. **Scaling:** min-max normalisation
5. **Train/test split:** 80/20
6. **Class balancing:** SMOTE applied to the training set only
7. **Model training:** custom BPNN
8. **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix

## Model Architecture
| Layer | Neurons | Activation |
|---|---|---|
| Input | 31 | – |
| Hidden 1 | 256 | ReLU |
| Hidden 2 | 128 | ReLU |
| Hidden 3 | 64 | ReLU |
| Output | 3 | Softmax |

- **Loss:** categorical cross-entropy
- **Optimiser:** full-batch gradient descent (learning rate 0.01, 1,500 epochs)
- **Weight initialisation:** He initialisation

## Results
**Test accuracy: 73.22%**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Graduate | 0.86 | 0.81 | 0.83 |
| Dropout | 0.81 | 0.70 | 0.75 |
| Enrolled | 0.39 | 0.57 | 0.46 |
| **Weighted avg** | **0.76** | **0.73** | **0.74** |

The model performs well on Graduate and Dropout. The Enrolled class is the hardest to predict because these students share characteristics with both other groups.

## Tech Stack
Python · NumPy · pandas · Matplotlib · imbalanced-learn (SMOTE) · scikit-learn (metrics only) · Google Colab

## How to Run
1. Download `data.csv` from the UCI link above
2. Open the notebook in Google Colab
3. Run the first cell and upload `data.csv` when prompted
4. Run all cells

## Limitations & Future Work
- Scaling and outlier clipping were applied before the train/test split. Fitting them on training data only would avoid slight data leakage.
- Test accuracy was monitored during training. A separate validation set would be better for early stopping.
- Future work: mini-batch training, dropout regularisation, and comparison with scikit-learn / Keras models.

## Team (ISP560 – UiTM Shah Alam)
- Nur Damia Izzati Zuhaimi
- Nurul Khausar Faqihah Mohd Suhaimi
- Alya Fariha Azairry
- Iman Mohamad Nasir
- Ahmad Afiq Mohd Khalil
