#!/usr/bin/env python3
"""
SymptomML: A Machine Learning Training Program for Disease Prediction
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Prevent TensorFlow warning spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('ggplot')
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("[WARNING] Visualization libraries not available. Visual outputs disabled.")
    VISUALIZATION_AVAILABLE = False

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Try importing deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    # Suppress TF logs
    tf.get_logger().setLevel('ERROR')
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# Define stopwords for text processing
STOPWORDS = {
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through',
    'during', 'before', 'after', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off'
}

def preprocess_text(text):
    """Process text by removing stopwords and punctuation"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS]
    return ' '.join(tokens)

def create_dataset():
    """Create a sample dataset of symptoms and diseases"""
    print("[INFO] Creating symptom-disease dataset...")
    os.makedirs('data', exist_ok=True)
    
    # Sample data with symptoms and corresponding diseases
    data = """Symptoms,Disease
fever headache cough,Common Cold
mild fever runny nose sore throat,Common Cold
cough congestion sneezing,Common Cold
headache sore throat mild fever,Common Cold
fatigue cough runny nose,Common Cold
high fever severe headache stiff neck light sensitivity,Meningitis
stiff neck fever vomiting confusion,Meningitis
headache fever stiff neck rash,Meningitis
light sensitivity headache fever neck pain,Meningitis
confusion fever stiff neck headache,Meningitis
chest pain shortness of breath sweating,Heart Attack
pain radiating to arm jaw neck,Heart Attack
chest pressure nausea cold sweat,Heart Attack
shortness of breath chest discomfort fatigue,Heart Attack
chest tightness dizziness anxiety,Heart Attack
fatigue weight loss night sweats cough,Tuberculosis
coughing blood chest pain fever,Tuberculosis
fatigue persistent cough weight loss,Tuberculosis
night sweats fever persistent cough,Tuberculosis
chest pain fatigue coughing blood,Tuberculosis
abdominal pain diarrhea nausea vomiting,Gastroenteritis
stomach cramps watery diarrhea,Gastroenteritis
nausea vomiting fever diarrhea,Gastroenteritis
abdominal pain fever vomiting,Gastroenteritis
diarrhea dehydration stomach pain,Gastroenteritis
high fever fatigue sore throat swollen lymph glands,Mononucleosis
swollen lymph nodes fatigue fever,Mononucleosis
extreme fatigue sore throat headache,Mononucleosis
fever swollen spleen fatigue,Mononucleosis
sore throat fever fatigue rash,Mononucleosis
fever rash joint pain muscle pain,Dengue
high fever headache pain behind eyes,Dengue
muscle joint pain rash vomiting,Dengue
fever rash fatigue bleeding gums,Dengue
severe headache pain behind eyes fever,Dengue
frequent urination excessive thirst hunger weight loss,Diabetes
increased thirst frequent urination fatigue,Diabetes
blurry vision slow healing wounds,Diabetes
weight loss extreme hunger fatigue,Diabetes
tingling hands feet excessive thirst,Diabetes
wheezing shortness of breath chest tightness coughing,Asthma
shortness of breath wheezing coughing,Asthma
chest tightness difficulty breathing wheezing,Asthma
coughing at night shortness of breath,Asthma
exercise induced breathing difficulty,Asthma"""
    
    with open('data/symptom_disease.csv', 'w') as f:
        f.write(data)
    
    # Load and preprocess the dataset
    df = pd.read_csv('data/symptom_disease.csv')
    df['Processed_Symptoms'] = df['Symptoms'].apply(preprocess_text)
    
    print(f"[SUCCESS] Dataset created with {len(df)} samples and {df['Disease'].nunique()} diseases")
    return df

def train_ml_models(X_train, X_test, y_train, y_test, class_names):
    """Train machine learning models and evaluate performance"""
    print("\n[INFO] Training ML models...")
    
    # Train Naive Bayes model
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    print(f"[RESULT] Naive Bayes accuracy: {nb_acc:.2%}")
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"[RESULT] Random Forest accuracy: {rf_acc:.2%}")
    
    # Print detailed classification report for the best model
    best_model_name = "Random Forest" if rf_acc > nb_acc else "Naive Bayes"
    best_preds = rf_pred if rf_acc > nb_acc else nb_pred
    
    print(f"\n[INFO] Best model: {best_model_name} ({max(rf_acc, nb_acc):.2%})")
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds, target_names=class_names, zero_division=0))
    
    # Create confusion matrix visualization
    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, best_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        print("[INFO] Confusion matrix saved as 'confusion_matrix.png'")
    
    return nb_model, rf_model

def train_dl_model(X_train, X_test, y_train, y_test, num_classes):
    """Train a deep learning model if TensorFlow is available"""
    if not DL_AVAILABLE:
        print("[WARNING] TensorFlow not available. Skipping deep learning model.")
        return None
    
    try:
        print("\n[INFO] Training deep learning model...")
        
        # Convert to categorical targets
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        
        # Convert sparse matrices to dense if needed
        if not isinstance(X_train, np.ndarray):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
        
        # Create model
        model = Sequential([
            Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train model with shorter epochs to avoid issues
        print("[INFO] Training neural network for 10 epochs...")
        history = model.fit(
            X_train, y_train_cat,
            epochs=10,
            batch_size=4,
            validation_split=0.2,
            verbose=0  # Suppress detailed output
        )
        
        # Evaluate model
        dl_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        dl_acc = accuracy_score(y_test, dl_pred)
        print(f"[RESULT] Deep Learning model accuracy: {dl_acc:.2%}")
        
        # Plot training history
        if VISUALIZATION_AVAILABLE:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('dl_training_history.png')
            plt.close()
            print("[INFO] DL training history saved as 'dl_training_history.png'")
        
        return model
    
    except Exception as e:
        print(f"[ERROR] Could not train deep learning model: {str(e)}")
        print("[INFO] Continuing with ML models only")
        return None

def visualize_feature_importance(vectorizer, rf_model):
    """Visualize feature importance from the Random Forest model"""
    if not VISUALIZATION_AVAILABLE:
        print("[INFO] Feature importance visualization skipped (visualization libraries not available)")
        return
    
    if hasattr(rf_model, 'feature_importances_'):
        feature_names = vectorizer.get_feature_names_out()
        feature_importance = rf_model.feature_importances_
        
        # Get top 20 features
        indices = np.argsort(feature_importance)[-20:]
        plt.figure(figsize=(10, 8))
        plt.title('Top 20 Important Features')
        plt.barh(range(len(indices)), feature_importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("[INFO] Feature importance plot saved as 'feature_importance.png'")
        
        print("\n[INFO] Top 10 important features:")
        for i in indices[-10:]:
            print(f"  - {feature_names[i]}: {feature_importance[i]:.4f}")

def predict_example(vectorizer, models, le, symptoms):
    """Make predictions on example symptoms"""
    print(f"\n[TEST] Prediction test with: '{symptoms}'")
    
    # Preprocess input
    processed = preprocess_text(symptoms)
    print(f"  Processed: '{processed}'")
    
    # Vectorize
    X = vectorizer.transform([processed])
    
    # Get predictions from each model
    results = []
    
    # Naive Bayes
    nb_model = models['nb']
    nb_proba = nb_model.predict_proba(X)[0]
    nb_top_idx = np.argsort(nb_proba)[-1]
    nb_top_disease = le.inverse_transform([nb_top_idx])[0]
    results.append((nb_top_disease, nb_proba[nb_top_idx], "NB"))
    
    # Random Forest
    rf_model = models['rf']
    rf_proba = rf_model.predict_proba(X)[0]
    rf_top_idx = np.argsort(rf_proba)[-1]
    rf_top_disease = le.inverse_transform([rf_top_idx])[0]
    results.append((rf_top_disease, rf_proba[rf_top_idx], "RF"))
    
    # Deep Learning
    if 'dl' in models and models['dl'] is not None:
        try:
            dl_model = models['dl']
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = X
            dl_proba = dl_model.predict(X_dense, verbose=0)[0]
            dl_top_idx = np.argmax(dl_proba)
            dl_top_disease = le.inverse_transform([dl_top_idx])[0]
            results.append((dl_top_disease, dl_proba[dl_top_idx], "DL"))
        except Exception as e:
            print(f"  [NOTE] DL prediction skipped: {str(e)}")
    
    # Print results
    print("\n  Prediction Results:")
    for disease, prob, model_name in results:
        print(f"    - {model_name}: {disease} ({prob:.2%})")

def main():
    """Main function to run the training program"""
    print("=" * 50)
    print("SYMPTOM ML: DISEASE PREDICTION TRAINING PROGRAM")
    print("=" * 50)
    
    # Create dataset
    df = create_dataset()
    
    # Plot disease distribution
    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(12, 6))
        disease_counts = df['Disease'].value_counts()
        sns.barplot(x=disease_counts.values, y=disease_counts.index)
        plt.title('Distribution of Diseases in Dataset')
        plt.xlabel('Number of Samples')
        plt.tight_layout()
        plt.savefig('disease_distribution.png')
        plt.close()
        print("[INFO] Disease distribution plot saved as 'disease_distribution.png'")
    
    # Prepare data for training
    print("\n[INFO] Preparing features and labels...")
    
    # Create feature vectors using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['Processed_Symptoms'])
    
    # Encode disease labels
    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
    print(f"[INFO] Feature dimension: {X_train.shape[1]}")
    
    # Train models
    nb_model, rf_model = train_ml_models(X_train, X_test, y_train, y_test, le.classes_)
    
    # Train deep learning model if available
    dl_model = train_dl_model(X_train, X_test, y_train, y_test, len(le.classes_))
    
    # Visualize feature importance
    visualize_feature_importance(vectorizer, rf_model)
    
    # Store models in a dictionary
    models = {'nb': nb_model, 'rf': rf_model}
    if dl_model:
        models['dl'] = dl_model
    
    # Test predictions with examples
    examples = [
        "fever headache cough",
        "chest pain shortness of breath sweating",
        "stiff neck fever headache confusion",
        "abdominal pain diarrhea nausea"
    ]
    
    for example in examples:
        predict_example(vectorizer, models, le, example)
    
    # Save trained models
    try:
        print("\n[INFO] Saving trained models...")
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
        joblib.dump(rf_model, 'models/random_forest_model.pkl')
        joblib.dump(le, 'models/label_encoder.pkl')
        if dl_model:
            try:
                dl_model.save('models/deep_learning_model')
            except Exception as e:
                print(f"[WARNING] Could not save deep learning model: {str(e)}")
        
        print("[SUCCESS] Models saved to 'models/' directory")
        print("\n[INFO] To use the models for prediction, load them with joblib and use them as follows:")
        print("""
    import joblib
    
    # Load models
    vectorizer = joblib.load('models/vectorizer.pkl')
    nb_model = joblib.load('models/naive_bayes_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    
    # Preprocess and vectorize symptoms
    symptoms = "fever headache cough"
    processed = preprocess_text(symptoms)
    X = vectorizer.transform([processed])
    
    # Predict with Random Forest model
    pred_idx = rf_model.predict(X)[0]
    predicted_disease = le.inverse_transform([pred_idx])[0]
    print(f"Predicted disease: {predicted_disease}")
    """)
    except Exception as e:
        print(f"[WARNING] Could not save models: {str(e)}")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
