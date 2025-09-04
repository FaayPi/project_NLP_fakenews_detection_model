# Model Development Pipeline: Word2Vec + XGBoost Classification

This document explains the complete machine learning pipeline implemented in `model_W2V_XGBoosting.py` for text classification using Word2Vec embeddings and XGBoost.

## 1. Load Data Including Duplicate Check

### Data Loading
- **Initial Data**: Loaded from `~/Downloads/data.csv`
- **Validation Data**: Loaded from `~/Downloads/validation_data.csv`
- **Format**: CSV files with columns: `label`, `title`, `text`, `subject`, `date`

### Duplicate Removal
- **Applied only to initial data** (not validation data)
- **Function**: `remove_duplicates(df, text_columns=['title', 'text'])`
- **Process**:
  1. Remove duplicates based on `title` column
  2. Remove duplicates based on `text` column
- **Rationale**: Validation data should not have rows deleted to maintain evaluation integrity

## 2. Preprocessing

### Combined Title-Text Feature
- **Function**: `add_combined_title_text_feature(df)`
- **Process**: Creates `title_text_combined` by concatenating title and text
- **Purpose**: Provides unified text for feature extraction

### Statistical Features
- **Uppercase Percentage**: `add_uppercase_percentage_feature(df)`
  - Calculates percentage of uppercase characters in combined text
  - Formula: `(uppercase_chars / total_chars) * 100`
- **Punctuation Count**: `add_punctuation_count_feature(df)`
  - Counts exclamation marks (!) and question marks (?)
- **URL Count**: `add_url_count_feature(df)`
  - Uses regex pattern: `r'https?://\S+|www\.\S+'`
  - Counts URLs in text
- **Mention Count**: `add_mention_count_feature(df)`
  - Uses regex pattern: `r'@\w+'`
  - Counts @mentions in text
- **Text Length Feature**
  - Function**: `add_text_length_feature(df)`
  - Process**: Creates `text_length` column using `df['text'].str.len()`
  - Purpose**: Captures document length as a numerical feature

## 3. Cleaning Including Deleting Features, Lowercase, etc.

### Comprehensive Text Cleaning
- **Function**: `clean_text_comprehensive(text)`
- **Steps**:
  1. **Lowercase conversion**: `text.lower()`
  2. **HTML tag removal**: `re.sub(r'<[^>]+>', '', text)`
  3. **URL removal**: `re.sub(r'https?://\S+|www\.\S+', '', text)`
  4. **Mention removal**: `re.sub(r'@\w+', '', text)`
  5. **Contraction resolution**: Expands contractions (e.g., "don't" → "do not")
  6. **Abbreviation normalization**: Expands abbreviations (e.g., "U.S." → "united states")
  7. **Number removal**: `re.sub(r'\d+', '', text)`
  8. **Special character removal**: `re.sub(r'[^\w\s]', '', text)`
  9. **Whitespace normalization**: `re.sub(r'\s+', ' ', text).strip()`

### Stop Word Removal
- **Primary**: Uses NLTK stopwords (`stopwords.words('english')`)
- **Fallback**: Basic stop word list if NLTK unavailable
- **Process**: Removes common words that don't carry significant meaning

### Original Column Removal
- **Function**: `remove_original_columns(df)`
- **Removed columns**: `subject`, `date`, `title`, `text`, `title_text_combined`
- **Rationale**: Keep only engineered features for model training

## 4. Lemmatization and Tokenization

### Lemmatization
- **Primary**: Uses NLTK WordNetLemmatizer
- **Fallback**: Simple rule-based lemmatization
  - Removes 'ing', 'ed', 's', 'ly' endings
- **Purpose**: Reduces words to their base form

### Tokenization and N-Grams
- **Function**: `add_tokenization_and_ngrams(df)`
- **Method**: CountVectorizer with N-gram features
- **Parameters**:
  - `ngram_range=(1, 2)`: 1-grams (single words) and 2-grams (word pairs)
  - `max_features=1000`: Top 1000 most frequent features
  - `min_df=2`: Minimum document frequency of 2
  - `max_df=0.95`: Maximum document frequency of 95%
- **Output**: Creates features like `ngram_word1`, `ngram_word2`, etc.

## 5. Feature Extraction with Word2Vec

### Word2Vec Model Training
- **Function**: `add_word2vec_features(df)`
- **Model**: `gensim.models.Word2Vec`
- **Parameters**:
  - `vector_size=100`: 100-dimensional word vectors
  - `window=5`: Context window size
  - `min_count=2`: Minimum word frequency
  - `workers=4`: Number of CPU cores
  - `sg=1`: Skip-gram algorithm (better for small datasets)
  - `epochs=10`: Training epochs
- **Training data**: Tokenized cleaned text from `title_text_cleaned`

### Document Vector Creation
- **Method**: Average of all word vectors in document
- **Function**: `get_document_vector(text, model)`
- **Process**:
  1. Split text into words
  2. Get Word2Vec vector for each word
  3. Calculate mean of all word vectors
  4. Return zero vector if no words found
- **Output**: 100 features named `w2v_0`, `w2v_1`, ..., `w2v_99`

## 6. Train-Test Split and Model Training

### Data Preparation
- **Features**: All columns except `label` and `title_text_cleaned`
- **Target**: `label` column
- **Split**: 80% training, 20% test
- **Method**: Stratified split to maintain class distribution

### XGBoost Model Configuration
- **Model**: `xgboost.XGBClassifier`
- **Parameters**:
  - `n_estimators=100`: Number of boosting rounds
  - `max_depth=6`: Maximum tree depth
  - `learning_rate=0.1`: Learning rate
  - `random_state=42`: For reproducibility
  - `eval_metric='logloss'`: Evaluation metric
  - `use_label_encoder=False`: Disable label encoding

### Training Process
- **Method**: `xgb_model.fit(X_train, y_train)`
- **Output**: Trained XGBoost classifier

## 7. Scores

### Model Evaluation Metrics
- **Training Accuracy**: Performance on training data
- **Test Accuracy**: Performance on held-out test data
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **Feature Importance**: Top 10 most important features

## 8. Validation

The trained model was used to predict labels for the validation dataset. The predicted values were added to the dataset `fp_3.csv` with the same structure as the original validation data.

### Validation Results
- **Accuracy**: 98.95%
- **Precision**: 96.53%
- **Recall**: 99.93%
- **F1 Score**: 98.20%

These excellent results demonstrate the model's strong performance on unseen data, with high precision and recall indicating balanced performance across both classes.

## Pipeline Summary

The complete pipeline transforms raw text data into numerical features using:
1. **Statistical features** (length, punctuation, etc.)
2. **N-gram features** (word and phrase frequencies)
3. **Word2Vec features** (semantic word embeddings)

These features are then used to train an XGBoost classifier, which achieves high accuracy through gradient boosting of decision trees. The model is evaluated on both test and validation datasets to ensure robust performance estimation.
