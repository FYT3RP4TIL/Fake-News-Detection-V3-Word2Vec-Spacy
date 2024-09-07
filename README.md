# 📰 Fake News Classification Project

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Dataset](#-dataset)
3. [Setup and Installation](#-setup-and-installation)
4. [Running the Notebook](#-running-the-notebook)
5. [Methodology](#️-methodology)
6. [Results](#-results)
7. [Key Takeaways](#-key-takeaways)
8. [Future Work](#-future-work)

## 🎯 Problem Statement

This project addresses the challenge of distinguishing between real and fake news articles using Natural Language Processing (NLP) techniques and machine learning algorithms. Our goal is to develop a classifier that can accurately identify fake news, contributing to the ongoing efforts to combat misinformation.

## 📊 Dataset

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **File**: "Fake_Real_Data.csv"
- **Columns**: Text (news content), Label (Fake or Real)
- **Size**: 9,900 entries (5,000 Fake, 4,900 Real)

## 🛠 Setup and Installation

To run this project, you'll need Python and Jupyter Notebook installed. Follow these steps:

1. Clone the repository or download the IPython notebook.

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install pandas numpy scikit-learn spacy jupyter
   ```

4. Download the spaCy model:
   ```
   python -m spacy download en_core_web_lg
   ```

5. Ensure you have the "Fake_Real_Data.csv" file in the same directory as the notebook.

## 🚀 Running the Notebook

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open the "Fake_News_Classification.ipynb" file in the Jupyter interface.

3. Run the cells in order, following the instructions within the notebook.

## 🛠️ Methodology

The notebook guides you through the following steps:

### 1. Data Loading and Exploration

```python
import pandas as pd

# Read the dataset
df = pd.read_csv("Fake_Real_Data.csv")

# Print the shape of dataframe
print(df.shape)

# Print top 5 rows
df.head(5)

# Check the distribution of labels 
df['label'].value_counts()
```

### 2. Text Vectorization

We use spaCy's `en_core_web_lg` model to create word embeddings:

```python
import spacy
nlp = spacy.load("en_core_web_lg")

# This will take some time (nearly 15 minutes)
df['vector'] = df['Text'].apply(lambda text: nlp(text).vector)
```

### 3. Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.vector.values,
    df.label_num,
    test_size=0.2,
    random_state=2022
)

import numpy as np

X_train_2d = np.stack(X_train) # converting to 2d numpy array
X_test_2d = np.stack(X_test)
```

### 4. Model Training and Evaluation

We implement and compare two models:

#### Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_train_embed = scaler.fit_transform(X_train_2d)
scaled_test_embed = scaler.transform(X_test_2d)

clf = MultinomialNB()
clf.fit(scaled_train_embed, y_train)

from sklearn.metrics import classification_report

y_pred = clf.predict(scaled_test_embed)
print(classification_report(y_test, y_pred))
```

#### K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
clf.fit(X_train_2d, y_train)

y_pred = clf.predict(X_test_2d)
print(classification_report(y_test, y_pred))
```

## 📊 Results

The notebook presents classification reports for both models:

### Multinomial Naive Bayes
```
             precision    recall  f1-score   support

           0       0.95      0.94      0.95      1024
           1       0.94      0.95      0.94       956

    accuracy                           0.94      1980
   macro avg       0.94      0.94      0.94      1980
weighted avg       0.94      0.94      0.94      1980
```

### K-Nearest Neighbors (KNN)
```
            precision    recall  f1-score   support

           0       1.00      0.99      0.99      1024
           1       0.99      0.99      0.99       956

    accuracy                           0.99      1980
   macro avg       0.99      0.99      0.99      1980
weighted avg       0.99      0.99      0.99      1980
```

## 🔑 Key Takeaways

1. **Effective Vectorization**: GloVe embeddings from spaCy provided rich 300-dimensional vectors, capturing semantic relationships effectively.

2. **Model Performance**: 
   - KNN achieved exceptional accuracy (99%), benefiting from the compact, semantic-rich GloVe vectors.
   - Multinomial Naive Bayes performed well (94% accuracy) after scaling to handle negative values.

3. **Preprocessing Impact**: Pre-trained GloVe embeddings significantly enhanced both models' performance, especially KNN.

4. **Time Consideration**: While GloVe embedding is time-consuming (about 15 minutes for this dataset), it results in high-quality feature representations.

## 🚀 Future Work

1. Experiment with deep learning models like LSTM or BERT.
2. Incorporate additional features (e.g., article source, publication date).
3. Develop a real-time classification system.
4. Explore explainable AI techniques for model interpretability.

---

📌 **Note**: This project is for educational purposes. Always critically evaluate news sources and cross-reference information, regardless of model predictions.
