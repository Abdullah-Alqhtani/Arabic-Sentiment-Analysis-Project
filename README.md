# Arabic-Sentiment-Analysis-Project
Arabic sentiment analysis project using preprocessing, N-gram models, and Naive Bayes, with improved performance using TF-IDF.
# Arabic Sentiment Analysis Project

## Overview

This project implements an Arabic sentiment analysis pipeline using:

* Text preprocessing for Arabic tweets
* N-gram Language Models (Bigram & Trigram)
* Naive Bayes classifier (from scratch)
* Bonus: Scikit-learn model using TF-IDF + MultinomialNB

The goal is to compare raw vs preprocessed text and evaluate different modeling approaches.

---

## Dataset

We used the dataset:
**Arabic Sentiment Twitter Corpus (arbml)**

Loaded via:

```python
from datasets import load_dataset
ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")
```

---

## Preprocessing

Implemented in `preprocessing.py`:

* Remove URLs and mentions
* Remove emojis and punctuation
* Normalize Arabic letters (Alef variants)
* Remove diacritics
* Reduce repeated characters
* Tokenization

### Example

Before:

```
@user أناااا سعيد جداً!!! 😊 #جميل
```

After:

```
انا سعيد جدا جميل
```

---

## Language Model

Implemented in `language_model.py`:

* Bigram and Trigram models
* Add-one smoothing
* Log probability computation
* Perplexity calculation
* Sentence generation

### Observations

* Trigram captures context better than Bigram
* Perplexity decreases with better preprocessing

---

## Naive Bayes Classifier

Implemented from scratch in `naive_bayes.py`:

* Uses Laplace smoothing
* Works on tokenized text
* Predicts sentiment labels (0 = negative, 1 = positive)

---

## Evaluation

Metrics implemented:

* Accuracy
* Precision
* Recall
* F1-score

---

## Results

### Custom Naive Bayes

* Accuracy: ~0.65 – 0.75

### Scikit-learn Model (TF-IDF + MultinomialNB)

* Accuracy: ~0.78
* F1-score: ~0.78

---

## Comparison

| Model                 | Accuracy |
| --------------------- | -------- |
| Raw Naive Bayes       | Lower    |
| Clean Naive Bayes     | Higher   |
| TF-IDF + NB (sklearn) | Best     |

---

## Key Insights

* Preprocessing significantly improves performance
* TF-IDF weighting enhances classification
* N-gram models capture context but increase complexity
* Errors often occur due to sarcasm or ambiguous sentiment

---

## How to Run

1. Install dependencies:

```bash
pip install datasets scikit-learn
```

2. Run the notebook or Python files step by step

---

## Project Structure

```
preprocessing.py
language_model.py
naive_bayes.py
evaluation.py
README.md
```

---

## Author

Name: [Abdullah]
