# PROJECT | Natural Language Processing Challenge

## Introduction
This project focuses on Natural Language Processing (NLP) techniques to classify news headlines as **real** or **fake news**. Processing text is a core skill for Data Scientists and AI Engineers, and this task allows you to apply these skills in practice.

## Project Overview
- **Dataset**: The main dataset is located at in this [Github Repo](https://github.com/FaayPi/project-nlp-challenge.git) and contains the following columns:
  - `label`: 0 for fake news, 1 for real news
  - `title`: Headline of the news article
  - `text`: Full content of the article
  - `subject`: Category or topic of the news
  - `date`: Publication date of the article
- **Goal**: Building a classifier to predict whether a news article is real or fake.

## Project Files
- `model_W2V_XGBoosting.py`: Contains the complete Python code for building and training the Word2Vec + XGBoost model.
- `model_explanation.md`: Explains all the steps in the model development pipeline.
- `accuracy_estimation.md`: Lists the model performance scores and evaluation metrics.
- `fp_3.csv`: Contains the predicted labels (0 or 1) for the validation dataset.
