Project Report
==============

Overview
--------

This document summarizes the progress, reasoning, and findings related to the Toxic Comment Classification project.

Baseline Model
--------------

- Used TF-IDF vectorization combined with Logistic Regression (OneVsRestClassifier).
- Achieved baseline ROC-AUC scores per class as follows:
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

