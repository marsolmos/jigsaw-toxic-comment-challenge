Project Report
==============

Overview
--------

This document summarizes the progress, reasoning, and findings related to the Toxic Comment Classification project. Our general approach follows the path taken by top-performing Kaggle competition winners. In addition to presenting detailed outcomes of the architectures and preprocessing techniques used, we highlight two main takeaways: preprocessing steps tend to add limited value, while leveraging embeddings consistently yields the best results. Based on these insights, we establish a simple baseline model, evaluate the impact of preprocessing, and experiment with embedding-based methods.

Baseline Model
--------------

First of all, we established a baseline model using the Kaggle Toxic Comment Classification Challenge dataset. The goal was to classify comments into six categories: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate.

We used the following approach:

- Used TF-IDF vectorization combined with Logistic Regression (OneVsRestClassifier).
- Achieved baseline ROC-AUC scores per class as follows:

+-------+--------------+---------+--------+--------+---------------+
| toxic | severe_toxic | obscene | threat | insult | identity_hate |
+=======+==============+=========+========+========+===============+
| 0.83  | 0.63         | 0.83    | 0.62   | 0.77   | 0.62          |
+-------+--------------+---------+--------+--------+---------------+

Updated Preprocessing
---------------------

To improve the baseline model, we updated the preprocessing steps. In this way, we aimed to enhance the model's performance by addressing issues like punctuation, special characters, and stop words. However, the findings from winners of the Kaggle competition indicated that preprocessing changes had limited impact on performance. Therefore, we focused on a minimal set of preprocessing steps, and expecting that the results would be similar to the baseline.

The updated preprocessing included:
- Remove HTML tags.
- Remove URLs.
- Lowercasing and stripping whitespace.
- Remove emails
- Replace multiple whitespaces with a single space.

The updated model achieved, as expected, similar ROC-AUC scores as the baseline, indicating that the preprocessing changes did not significantly impact performance. Achieved ROC-AUC scores per class were:

+-------+--------------+---------+--------+--------+---------------+
| toxic | severe_toxic | obscene | threat | insult | identity_hate |
+=======+==============+=========+========+========+===============+
| 0.83  | 0.63         | 0.83    | 0.62   | 0.77   | 0.62          |
+-------+--------------+---------+--------+--------+---------------+

Embedding-Based Models
----------------------

To further improve the model, we experimented with embedding-based methods. We used pre-trained embeddings from Sentence Transformers, which have shown to be effective in various NLP tasks.

We implemented the following embedding-based models:

+-------+--------------+---------+--------+--------+---------------+
| toxic | severe_toxic | obscene | threat | insult | identity_hate |
+=======+==============+=========+========+========+===============+
| 0.96  | 0.984        | 0.975   | 0.964  | 0.972  | 0.98          |
+-------+--------------+---------+--------+--------+---------------+

Conclusion
----------

The project demonstrated that while preprocessing steps have limited impact on performance, leveraging embeddings consistently yields significant improvements. The embedding-based models outperformed the baseline and updated preprocessing models, achieving ROC-AUC scores close to 0.98 across all classes.
This aligns with the findings from the Kaggle competition winners, who emphasized the importance of embeddings in achieving high performance in toxic comment classification tasks.

As next steps, we plan to explore more advanced embedding techniques, such as fine-tuning pre-trained models on the specific dataset, and potentially integrating additional features or models to further enhance performance, like:
- Embeddings + Light fine-tuning: Fine-tuning the pre-trained embeddings on the specific dataset to adapt them to the task.
- Embeddings + LSTM: Using embeddings as input to an LSTM model to capture sequential dependencies in the comments.
- Increasing training data: Augmenting the dataset with additional toxic comments to improve model robustness. For example, translating comments to different languages and back to English, or using data augmentation techniques like synonym replacement or random insertion.
- Transformers: Exploring transformer-based models like BERT or RoBERTa, which have shown state-of-the-art performance in various NLP tasks, including toxic comment classification.

Apart from the ML models, we also plan to do some general improvements in the project, like:
- Enhance project versioning: Implementing a more structured versioning system for the project, including model versions, preprocessing steps, and evaluation metrics.
- Documentation deployment: Deploy documentation to a GitHub Pages site or similar, to make it easily accessible for future reference and collaboration.
- API enhancements and deployment: Consider deploying the with FastAPI + Docker to make the model accessible via an API, allowing for easy integration with other applications or services.
- Model monitoring: Deploy MLFlow instance to monitor model performance in production, track metrics, and manage model versions.

References
----------

- Kaggle Toxic Comment Classification Challenge: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Kaggle Toxic Comment Classification Winners: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557
- Sentence Transformers: https://www.sbert.net/

