Training Scripts
================

This section describes the different training approaches implemented in the project.

.. contents::
   :local:
   :depth: 1

TF-IDF Baseline
---------------

File: ``src/training/baseline_tfidf.py``

- Uses a simple text vectorization pipeline with TF-IDF.
- Trains a Logistic Regression classifier using OneVsRest strategy.
- Serves as a starting point for performance comparison.

Improved Preprocessing
----------------------

File: ``src/training/improved_preprocessing.py``

- Applies additional text cleaning steps:
  - HTML stripping
  - URL/email removal
  - Whitespace normalization
- Uses the same model pipeline
