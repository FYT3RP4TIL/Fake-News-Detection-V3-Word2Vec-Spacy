﻿# 📰 Fake-News-Detection-V3-Word2Vec-Spacy

## 🎯 Problem Statement

Fake news is a growing concern in our digital age, spreading misinformation rapidly through various channels like social media and messaging apps. This project aims to address this issue by developing a classifier to distinguish between real and fake news using Natural Language Processing (NLP) techniques.

## 📊 Dataset

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Format**: CSV file named "Fake_Real_Data.csv"
- **Columns**: 
  - Text: The news content
  - Label: "Fake" or "Real"
- **Size**: 9,900 entries (5,000 Fake, 4,900 Real)

## 🛠️ Methodology

### 1. Data Preprocessing

We use the `spacy` library with the `en_core_web_lg` model to create word embeddings:

```python
import spacy
nlp = spacy.load("en_core_web_lg")
df['vector'] = df['Text'].apply(lambda text: nlp(text).vector)
```

This step creates a 300-dimensional vector for each news article, capturing semantic information.

### 2. Train-Test Split

We split the data into training (80%) and testing (20%) sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.vector.values,
    df.label_num,
    test_size=0.2,
    random_state=2022
)
```

### 3. Model Training and Evaluation

We trained two models:

#### 📊 Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_train_embed = scaler.fit_transform(X_train_2d)
scaled_test_embed = scaler.transform(X_test_2d)

clf = MultinomialNB()
clf.fit(scaled_train_embed, y_train)
```

**Classification Report:**

```
             precision    recall  f1-score   support

           0       0.95      0.94      0.95      1024
           1       0.94      0.95      0.94       956

    accuracy                           0.94      1980
   macro avg       0.94      0.94      0.94      1980
weighted avg       0.94      0.94      0.94      1980
```

#### 🏘️ K-Nearest Neighbors (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
clf.fit(X_train_2d, y_train)
```

**Classification Report:**

```
            precision    recall  f1-score   support

           0       1.00      0.99      0.99      1024
           1       0.99      0.99      0.99       956

    accuracy                           0.99      1980
   macro avg       0.99      0.99      0.99      1980
weighted avg       0.99      0.99      0.99      1980
```

## 🔑 Key Takeaways

1. **Vector Creation**: Using GloVe embeddings from spaCy resulted in 300-dimensional vectors for each news article. These embeddings capture semantic relationships between words, allowing for more nuanced classification.

2. **Model Performance**: 
   - The KNN model performed exceptionally well, achieving 99% accuracy. This is likely due to the reduced dimensionality (300) of the GloVe vectors compared to traditional methods like Bag-of-Words or TF-IDF.
   - Multinomial Naive Bayes also performed well (94% accuracy) after scaling the vectors to handle negative values.

3. **Preprocessing Impact**: The use of pre-trained GloVe embeddings significantly improved the performance of both models, especially KNN, which typically struggles with high-dimensional data.

4. **Scalability**: While the GloVe embedding process is time-consuming (about 15 minutes for this dataset), it results in more compact and semantically rich representations of the text data.

## 🚀 Future Work

1. Experiment with other deep learning models like LSTM or BERT for potentially better performance.
2. Incorporate additional features such as article source, publication date, or author credibility.
3. Develop a real-time classification system for incoming news articles.
4. Explore explainable AI techniques to understand what features contribute most to the classification decisions.

---

📌 **Note**: Always critically evaluate news sources and cross-reference information, regardless of model predictions. This classifier is a tool to assist in identifying potential fake news, not a definitive arbiter of truth.
