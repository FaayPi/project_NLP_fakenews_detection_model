import pandas as pd
import numpy as np
import os

# Load initial data from Downloads folder
print("Loading initial data from Downloads...")
download_path = os.path.expanduser("~/Downloads")
initial_data = pd.read_csv(os.path.join(download_path, 'data.csv'))

# Load validation data from Downloads folder
print("Loading validation data from Downloads...")
validation_data = pd.read_csv(os.path.join(download_path, 'validation_data.csv'))

print(f"Initial data loaded: {initial_data.shape}")
print(f"Validation data loaded: {validation_data.shape}")
print(f"\nColumns of initial data: {list(initial_data.columns)}")

# ===== PREPROCESSING PIPELINE =====

def remove_duplicates(df, text_columns=['title', 'text']):
    """
    Removes duplicates based on text columns
    """
    original_len = len(df)
    print(f"Original number of rows: {original_len}")
    
    # Remove duplicates based on title
    if 'title' in df.columns:
        df = df.drop_duplicates(subset=['title'], keep='first')
        print(f"After title duplicate removal: {len(df)}")
    
    # Remove duplicates based on text
    if 'text' in df.columns:
        df = df.drop_duplicates(subset=['text'], keep='first')
        print(f"After text duplicate removal: {len(df)}")
    
    removed_duplicates = original_len - len(df)
    print(f"Final number of rows: {len(df)}")
    print(f"Removed duplicates: {removed_duplicates}")
    
    return df

def add_text_length_feature(df):
    """
    Creates a column for text length
    """
    print("\n=== Creating text length column ===")
    df['text_length'] = df['text'].str.len()
    
    print(f"Text length column created!")
    print(f"Text length statistics:")
    print(df['text_length'].describe())
    
    return df

def add_combined_title_text_feature(df):
    """
    Creates a combined title+text column
    """
    print("\n=== Creating combined title+text column ===")
    df['title_text_combined'] = df['title'] + ' ' + df['text']
    
    print(f"Combined title+text column created!")
    print(f"Example of first combined line:")
    print(df['title_text_combined'].iloc[0][:200] + "...")
    
    return df

def add_uppercase_percentage_feature(df):
    """
    Creates a column for the percentage of uppercase characters in the combined column
    """
    print("\n=== Creating uppercase percentage column ===")
    
    def calculate_uppercase_percentage(text):
        if pd.isna(text) or text == '':
            return 0.0
        total_chars = len(text)
        uppercase_chars = sum(1 for char in text if char.isupper())
        return (uppercase_chars / total_chars) * 100 if total_chars > 0 else 0.0
    
    df['uppercase_percentage'] = df['title_text_combined'].apply(calculate_uppercase_percentage)
    
    print(f"Uppercase percentage column created!")
    print(f"Uppercase percentage statistics:")
    print(df['uppercase_percentage'].describe())
    
    return df

def add_punctuation_count_feature(df):
    """
    Creates a column for the count of exclamation and question marks
    """
    print("\n=== Creating punctuation count column ===")
    
    def count_punctuation(text):
        if pd.isna(text):
            return 0
        return text.count('!') + text.count('?')
    
    df['punctuation_count'] = df['title_text_combined'].apply(count_punctuation)
    
    print(f"Punctuation count column created!")
    print(f"Punctuation count statistics:")
    print(df['punctuation_count'].describe())
    
    return df

def add_url_count_feature(df):
    """
    Creates a column for the count of URLs
    """
    print("\n=== Creating URL count column ===")
    
    import re
    
    def count_urls(text):
        if pd.isna(text):
            return 0
        # URL pattern (simple)
        url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        return len(url_pattern.findall(text))
    
    df['url_count'] = df['title_text_combined'].apply(count_urls)
    
    print(f"URL count column created!")
    print(f"URL count statistics:")
    print(df['url_count'].describe())
    
    return df

def add_mention_count_feature(df):
    """
    Creates a column for the count of mentions (@username)
    """
    print("\n=== Creating mention count column ===")
    
    import re
    
    def count_mentions(text):
        if pd.isna(text):
            return 0
        # Mention pattern (@username)
        mention_pattern = re.compile(r'@\w+')
        return len(mention_pattern.findall(text))
    
    df['mention_count'] = df['title_text_combined'].apply(count_mentions)
    
    print(f"Mention count column created!")
    print(f"Mention count statistics:")
    print(df['mention_count'].describe())
    
    return df

def clean_text_comprehensive(text):
    """
    Comprehensive text cleaning for the title_text_combined column
    """
    if pd.isna(text):
        return ""
    
    import re
    import string
    
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    
    # 4. Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # 5. Resolve contractions
    contractions = {
        "don't": "do not", "can't": "cannot", "won't": "will not", "n't": " not",
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "i'll": "i will", "you'll": "you will", "he'll": "he will",
        "i'd": "i would", "you'd": "you would", "he'd": "he would"
    }
    for contraction, expansion in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    
    # 6. Normalize abbreviations
    abbreviations = {
        r'\bu\.s\.\b': 'united states', r'\bus\b': 'united states',
        r'\bdr\.\b': 'doctor', r'\bmr\.\b': 'mister', r'\bmrs\.\b': 'missus',
        r'\bvs\.\b': 'versus', r'\betc\.\b': 'etcetera'
    }
    for abbrev, full in abbreviations.items():
        text = re.sub(abbrev, full, text, flags=re.IGNORECASE)
    
    # 7. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 8. Remove special characters (except whitespace)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 10. Remove stop words
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    except:
        # Fallback if NLTK is not available
        basic_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = text.split()
        words = [word for word in words if word not in basic_stop_words]
        text = ' '.join(words)
    
    # 11. Lemmatization
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(lemmatized_words)
    except:
        # Fallback if NLTK is not available - simple lemmatization
        words = text.split()
        # Simple rules for common endings
        lemmatized_words = []
        for word in words:
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s'):
                word = word[:-1]
            elif word.endswith('ly'):
                word = word[:-2]
            lemmatized_words.append(word)
        text = ' '.join(lemmatized_words)
    
    return text

def add_cleaned_text_feature(df):
    """
    Creates a cleaned version of the title_text_combined column
    """
    print("\n=== Creating cleaned text column ===")
    
    df['title_text_cleaned'] = df['title_text_combined'].apply(clean_text_comprehensive)
    
    print(f"Cleaned text column created!")
    print(f"Example of first cleaned line:")
    print(df['title_text_cleaned'].iloc[0][:200] + "...")
    
    # Show length comparison
    original_length = df['title_text_combined'].str.len().mean()
    cleaned_length = df['title_text_cleaned'].str.len().mean()
    print(f"\nAverage length:")
    print(f"Original: {original_length:.1f} characters")
    print(f"Cleaned: {cleaned_length:.1f} characters")
    print(f"Reduction: {((original_length - cleaned_length) / original_length * 100):.1f}%")
    
    return df

def add_tokenization_and_ngrams(df):
    """
    Applies tokenization and N-grams to the cleaned text column
    """
    print("\n=== Creating tokenization and N-grams ===")
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Tokenization and N-grams (1-grams = single words, 2-grams = word pairs)
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),  # 1-grams and 2-grams
        max_features=1000,   # Top 1000 features
        min_df=2,           # At least 2 documents
        max_df=0.95         # Maximum 95% of documents
    )
    
    # Vectorization of cleaned texts
    X_ngrams = vectorizer.fit_transform(df['title_text_cleaned'])
    
    # Convert to DataFrame for better overview
    feature_names = vectorizer.get_feature_names_out()
    df_ngrams = pd.DataFrame(X_ngrams.toarray(), columns=feature_names)
    
    # Add N-gram features to the original DataFrame
    for col in df_ngrams.columns:
        df[f'ngram_{col}'] = df_ngrams[col]
    
    print(f"N-gram vectorization completed!")
    print(f"Number of N-gram features: {len(feature_names)}")
    print(f"Feature names (first 10): {feature_names[:10]}")
    
    # Show statistics of N-gram features
    ngram_columns = [col for col in df.columns if col.startswith('ngram_')]
    print(f"\nN-gram features statistics:")
    print(f"Number of features with values > 0:")
    print(df[ngram_columns].astype(bool).sum().describe())
    
    return df

def add_word2vec_features(df):
    """
    Applies Word2Vec to the cleaned text column
    """
    print("\n=== Creating Word2Vec features ===")
    
    from gensim.models import Word2Vec
    import numpy as np
    
    # Tokenize texts for Word2Vec
    tokenized_texts = [text.split() for text in df['title_text_cleaned']]
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,      # 100-dimensional vectors
        window=5,            # Context window
        min_count=2,         # Minimum frequency
        workers=4,           # Number of workers
        sg=1,                # Skip-gram (better for small datasets)
        epochs=10           # Number of epochs
    )
    
    print(f"Word2Vec model trained!")
    print(f"Vocabulary size: {len(w2v_model.wv.key_to_index)}")
    print(f"Vector dimension: {w2v_model.vector_size}")
    
    # Create document vectors (average of all word vectors per document)
    def get_document_vector(text, model):
        words = text.split()
        word_vectors = []
        for word in words:
            if word in model.wv.key_to_index:
                word_vectors.append(model.wv[word])
        
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    # Create Word2Vec features for each document
    w2v_features = []
    for text in df['title_text_cleaned']:
        doc_vector = get_document_vector(text, w2v_model)
        w2v_features.append(doc_vector)
    
    # Convert to DataFrame
    w2v_df = pd.DataFrame(w2v_features, columns=[f'w2v_{i}' for i in range(w2v_model.vector_size)])
    
    # Add Word2Vec features to the original DataFrame
    for col in w2v_df.columns:
        df[col] = w2v_df[col]
    
    print(f"Word2Vec features created!")
    print(f"Number of Word2Vec features: {w2v_model.vector_size}")
    print(f"Feature names: {list(w2v_df.columns)}")
    
    # Show statistics of Word2Vec features
    w2v_columns = [col for col in df.columns if col.startswith('w2v_')]
    print(f"\nWord2Vec features statistics:")
    print(df[w2v_columns].describe())
    
    return df

def remove_original_columns(df):
    """
    Removes the original columns and keeps only the features
    """
    print("\n=== Removing original columns ===")
    
    columns_to_remove = ['subject', 'date', 'title', 'text', 'title_text_combined']
    columns_before = list(df.columns)
    
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed: {col}")
        else:
            print(f"Column not found: {col}")
    
    columns_after = list(df.columns)
    print(f"\nColumns before cleaning: {columns_before}")
    print(f"Columns after cleaning: {columns_after}")
    print(f"Removed columns: {[col for col in columns_before if col not in columns_after]}")
    
    return df

def apply_preprocessing_pipeline(df):
    """
    Applies the complete preprocessing pipeline
    """
    print("=== STARTING PREPROCESSING PIPELINE ===")

    # Step 1: Remove duplicates
    print("\n--- Step 1: Remove duplicates ---")
    df = remove_duplicates(df)

    # Step 2: Add text length feature
    print("\n--- Step 2: Add text length feature ---")
    df = add_text_length_feature(df)

    # Step 3: Add combined title+text column
    print("\n--- Step 3: Add combined title+text column ---")
    df = add_combined_title_text_feature(df)

    # Step 4: Add uppercase percentage
    print("\n--- Step 4: Add uppercase percentage ---")
    df = add_uppercase_percentage_feature(df)

    # Step 5: Add punctuation count
    print("\n--- Step 5: Add punctuation count ---")
    df = add_punctuation_count_feature(df)

    # Step 6: Add URL count
    print("\n--- Step 6: Add URL count ---")
    df = add_url_count_feature(df)

    # Step 7: Add mention count
    print("\n--- Step 7: Add mention count ---")
    df = add_mention_count_feature(df)

    # Step 8: Add comprehensive text cleaning with lemmatization
    print("\n--- Step 8: Add comprehensive text cleaning with lemmatization ---")
    df = add_cleaned_text_feature(df)

    # Step 9: Add tokenization and N-grams
    print("\n--- Step 9: Add tokenization and N-grams ---")
    df = add_tokenization_and_ngrams(df)

    # Step 10: Remove original columns
    print("\n--- Step 10: Remove original columns ---")
    df = remove_original_columns(df)

    print(f"\n=== PREPROCESSING PIPELINE COMPLETED ===")
    print(f"Final dataset size: {df.shape}")

    return df

def apply_word2vec_pipeline(df):
    """
    Applies the Word2Vec pipeline
    """
    print("\n=== STARTING WORD2VEC PIPELINE ===")
    
    # Add Word2Vec features
    print("\n--- Adding Word2Vec features ---")
    df = add_word2vec_features(df)
    
    print(f"\n=== WORD2VEC PIPELINE COMPLETED ===")
    print(f"Final dataset size: {df.shape}")
    
    return df

def apply_model_training_pipeline(df):
    """
    Applies the model training pipeline
    """
    print("\n=== STARTING MODEL TRAINING PIPELINE ===")
    
    # Step 1: Train-Test-Split
    print("\n--- Step 1: Train-Test-Split ---")
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df.drop(columns=['label', 'title_text_cleaned'])
    y = df['label']
    
    # Train-Test-Split (80% Training, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training Set: {X_train.shape[0]} Samples")
    print(f"Test Set: {X_test.shape[0]} Samples")
    print(f"Features: {X_train.shape[1]} Columns")
    print(f"Target distribution Training: {y_train.value_counts().to_dict()}")
    print(f"Target distribution Test: {y_test.value_counts().to_dict()}")
    
    # Step 2: XGBoost Training
    print("\n--- Step 2: XGBoost Training ---")
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Initialize XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train model
    print("Training XGBoost model...")
    xgb_model.fit(X_train, y_train)
    
    # Step 3: Model Evaluation
    print("\n--- Step 3: Model Evaluation ---")
    
    # Predictions
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Feature Importance
    print("\nTop 10 Feature Importance:")
    feature_importance = xgb_model.feature_importances_
    feature_names = X_train.columns
    
    # Sort features by importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_df.head(10))
    
    # Step 4: Save model
    print("\n--- Step 4: Save model ---")
    import pickle
    
    # Save the trained model
    model_filename = 'xgboost_model_new.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save the train-test data
    data_filename = 'train_test_data_new.pkl'
    with open(data_filename, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }, f)
    
    print(f"Model saved: {model_filename}")
    print(f"Data saved: {data_filename}")
    
    print(f"\n=== MODEL TRAINING PIPELINE COMPLETED ===")
    
    return {
        'model': xgb_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

def apply_validation_pipeline(validation_data, training_results):
    """
    Applies the trained model to validation data
    """
    print("\n=== STARTING VALIDATION PIPELINE ===")
    
    # Step 1: Apply preprocessing to validation data
    print("\n--- Step 1: Apply preprocessing to validation data ---")
    validation_data_cleaned = apply_preprocessing_pipeline(validation_data.copy())
    
    # Step 2: Apply Word2Vec to validation data
    print("\n--- Step 2: Apply Word2Vec to validation data ---")
    validation_data_with_w2v = apply_word2vec_pipeline(validation_data_cleaned.copy())
    
    # Step 3: Prepare features for validation
    print("\n--- Step 3: Prepare features ---")
    
    # Use the same features as in training
    X_train = training_results['X_train']
    validation_features = validation_data_with_w2v.drop(columns=['label', 'title_text_cleaned'])
    
    # Ensure all features are present
    missing_features = set(X_train.columns) - set(validation_features.columns)
    extra_features = set(validation_features.columns) - set(X_train.columns)
    
    if missing_features:
        print(f"Missing features in validation data: {len(missing_features)}")
        for feature in missing_features:
            validation_features[feature] = 0
    
    if extra_features:
        print(f"Extra features in validation data removed: {len(extra_features)}")
        validation_features = validation_features[X_train.columns]
    
    print(f"Validation data features: {validation_features.shape}")
    
    # Step 4: Make predictions on validation data
    print("\n--- Step 4: Make predictions on validation data ---")
    xgb_model = training_results['model']
    
    # Predictions
    validation_predictions = xgb_model.predict(validation_features)
    validation_probabilities = xgb_model.predict_proba(validation_features)
    
    # Step 5: Save results
    print("\n--- Step 5: Save results ---")
    
    # Create result DataFrame
    validation_results_df = validation_data_with_w2v[['label', 'title_text_cleaned']].copy()
    validation_results_df['predicted_label'] = validation_predictions
    validation_results_df['prediction_probability'] = validation_probabilities.max(axis=1)
    validation_results_df['prediction_confidence'] = validation_probabilities.max(axis=1)
    
    # Save results
    validation_filename = 'validation_predictions_new.csv'
    validation_results_df.to_csv(validation_filename, index=False)
    
    print(f"Validation results saved: {validation_filename}")
    
    # Step 6: Validation metrics
    print("\n--- Step 6: Validation metrics ---")
    
    if 'label' in validation_data_with_w2v.columns:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        y_val_true = validation_data_with_w2v['label']
        val_accuracy = accuracy_score(y_val_true, validation_predictions)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Number of predictions: {len(validation_predictions)}")
        print(f"Prediction distribution: {pd.Series(validation_predictions).value_counts().to_dict()}")
        
        if len(y_val_true.unique()) > 1:
            print("\nClassification Report (Validation):")
            print(classification_report(y_val_true, validation_predictions))
            
            print("\nConfusion Matrix (Validation):")
            cm_val = confusion_matrix(y_val_true, validation_predictions)
            print(cm_val)
    else:
        print("No true labels available in validation data")
        print(f"Number of predictions: {len(validation_predictions)}")
        print(f"Prediction distribution: {pd.Series(validation_predictions).value_counts().to_dict()}")
    
    print(f"\n=== VALIDATION PIPELINE COMPLETED ===")
    
    return {
        'predictions': validation_predictions,
        'probabilities': validation_probabilities,
        'results_df': validation_results_df,
        'validation_accuracy': val_accuracy if 'label' in validation_data_with_w2v.columns else None
    }

# Apply preprocessing pipeline to initial data
initial_data_cleaned = apply_preprocessing_pipeline(initial_data.copy())

# Apply Word2Vec pipeline to cleaned data
initial_data_with_w2v = apply_word2vec_pipeline(initial_data_cleaned.copy())

# Apply model training pipeline
training_results = apply_model_training_pipeline(initial_data_with_w2v.copy())

# Apply validation pipeline
validation_results = apply_validation_pipeline(validation_data.copy(), training_results)

print("\n=== FINAL RESULTS ===")
print(f"Dataset after preprocessing: {initial_data_cleaned.shape}")
print(f"Dataset after Word2Vec: {initial_data_with_w2v.shape}")
print(f"Training Accuracy: {training_results['train_accuracy']:.4f}")
print(f"Test Accuracy: {training_results['test_accuracy']:.4f}")
if validation_results['validation_accuracy']:
    print(f"Validation Accuracy: {validation_results['validation_accuracy']:.4f}")

print("\nExample of final data (first 5 rows):")
print(initial_data_with_w2v.head())
