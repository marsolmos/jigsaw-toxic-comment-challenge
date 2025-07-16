Project Report
==============

Overview
--------

This document summarizes the progress, reasoning, and findings related to the Toxic Comment Classification project.

Baseline Model
--------------

First of all, we established a baseline model using the Kaggle Toxic Comment Classification Challenge dataset. The goal was to classify comments into six categories: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate.

We used the following approach:

- Used TF-IDF vectorization combined with Logistic Regression (OneVsRestClassifier).
- Achieved baseline ROC-AUC scores per class as follows:
  - Toxic: 0.83
  - Severe Toxic: 0.63
  - Obscene: 0.83
  - Threat: 0.62
  - Insult: 0.77
  - Identity Hate: 0.62

Updated Preprocessing
---------------------

To improve the baseline model, we updated the preprocessing steps. In this way, we aimed to enhance the model's performance by addressing issues like punctuation, special characters, and stop words.

The updated preprocessing included:
- Remove HTML tags.
- Remove URLs.
- Lowercasing and stripping whitespace.
- Remove emails
- Replace multiple whitespaces with a single space.

The updated model achieved similar ROC-AUC scores as the baseline, indicating that the preprocessing changes did not significantly impact performance. Achieved ROC-AUC scores per class were:
  - Toxic: 0.83
  - Severe Toxic: 0.63
  - Obscene: 0.83
  - Threat: 0.62
  - Insult: 0.77
  - Identity Hate: 0.62

References
----------

- Kaggle Toxic Comment Classification Challenge: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Sentence Transformers: https://www.sbert.net/

