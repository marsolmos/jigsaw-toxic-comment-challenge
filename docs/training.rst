Training Scripts
================

This section describes the different training approaches implemented in the project.

TF-IDF Baseline
---------------

File: ``src/training/train_baseline.py``

- Uses a simple text vectorization pipeline with TF-IDF.
- Trains a Logistic Regression classifier using OneVsRest strategy.
- Serves as a starting point for performance comparison.

Improved Preprocessing
----------------------

File: ``src/training/train_updated_preprocess.py``

- Applies additional text cleaning steps:
  - HTML stripping
  - URL/email removal
  - Whitespace normalization
- Uses the same model pipeline

Embedding Model
---------------

File: ``src/training/train_embedding_model.py``

- Introduces an embedding model for text representation.
- Uses a pre-trained embedding model to convert text into dense vectors.
- Trains a classifier on these embeddings.
- Aims to improve performance by capturing semantic meaning.
- This model can be used in conjunction with the improved preprocessing steps (not implemented as it didn't yield significant improvements).
