import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
import pickle
from sklearn.neighbors import KNeighborsClassifier 



def find_optimal_threshold(model, X_test, y_test):
    # 1. Get raw probabilities
    y_probs = model.predict(X_test, verbose=0)
    
    # 2. Setup storage
    thresholds = np.arange(0.0, 1.0, 0.005) # Test every 0.005 step
    accuracies = []
    
    best_threshold = 0.5
    best_acc = 0.0
    
    # 3. Loop through all candidates
    for thresh in thresholds:
        # Create predictions for this specific threshold
        y_pred = (y_probs > thresh).astype(int)
        
        # Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        # Save if it's the new record
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
            
    # 4. Plot the curve
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, accuracies, color='blue', label='Accuracy Curve')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.4f}')
    
    plt.title('Accuracy vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 5. Report
    print(f"--- Optimization Results ---")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Max Accuracy:   {best_acc*100:.2f}%")
    
    return best_threshold



def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, use_class_weights=True):
    """
    A flexible trainer that handles class weights and fitting for any Keras model.
    """
    
    # --- 1. Calculate Class Weights (Optional) ---
    weights_dict = None
    if use_class_weights:
        # Calculate weights based on the training data distribution
        cw = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        # Convert to dictionary format: {0: weight_0, 1: weight_1}
        weights_dict = {i: cw[i] for i in range(len(cw))}
        
        print(f"\n--- Computed Class Weights ---")
        for cls, weight in weights_dict.items():
            print(f"Class {cls}: {weight:.4f}")
    else:
        print("\n--- Class Weights: Disabled (1.0 for all) ---")

    # --- 2. Train the Model ---
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=weights_dict,  # Pass the dict or None
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # --- 3. Evaluate ---
    print("\n--- Final Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    
    return history, model



# LSTM Model

def build_lstm_model(input_shape):
    """
    Builds a standard LSTM model for binary classification.
    Compatible with the standardized train_model pipeline.
    
    Args:
        input_shape: tuple (time_steps, features) -> e.g. (14, 8)
    Returns:
        model: A compiled Keras Sequential model
    """
    model = Sequential()
    
    # --- Layer 1: LSTM ---
    # units=50: The number of "memory cells" or neurons.
    # return_sequences=False: We only need the output of the *last* day to make a prediction.
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    
    # --- Layer 2: Dropout (Regularization) ---
    # Randomly sets 20% of neurons to 0 during training.
    # CRITICAL for financial data to prevent memorizing noise.
    model.add(Dropout(0.2))
    
    # --- Layer 3: Output ---
    # units=1: We want a single probability score.
    # activation='sigmoid': Squashes output between 0 and 1.
    model.add(Dense(units=1, activation='sigmoid'))
    
    # --- Compilation ---
    # Optimizer: Adam is the standard "smart" gradient descent.
    # Loss: binary_crossentropy is required for Up/Down classification.
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model



# XGBoost Model

class XGBoostKerasAdapter:
    """
    A wrapper that makes XGBoost look exactly like a Keras model.
    Compatible with the standardized train_model pipeline.
    """
    def __init__(self, input_shape, learning_rate=0.05, n_estimators=100):
        self.input_shape = input_shape
        # Initialize the actual XGBoost Classifier
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,             # Depth 6 is standard for tabular data
            subsample=0.8,           # prevent overfitting
            colsample_bytree=0.8,    # prevent overfitting
            eval_metric='logloss',   # Metric to watch
            early_stopping_rounds=10,
            n_jobs=-1                # Use all CPU cores
        )
        self.history = None

    def _flatten(self, X):
        """Converts 3D LSTM input (Samples, Time, Feat) to 2D (Samples, Time*Feat)"""
        if len(X.shape) == 3:
            # Flatten: (1000, 14, 8) -> (1000, 112)
            return X.reshape(X.shape[0], -1)
        return X

    def compile(self, optimizer=None, loss=None, metrics=None):
        # Keras needs this method, XGBoost does not. We pass silently.
        pass

    def fit(self, X_train, y_train, epochs=None, batch_size=None, 
            validation_data=None, verbose=1, class_weight=None):
        """
        Mimics Keras .fit() but runs XGBoost training.
        """
        # 1. Flatten Data (XGBoost needs 2D)
        X_train_flat = self._flatten(X_train)
        
        # 2. Handle Validation Data
        eval_set = []
        if validation_data:
            X_val, y_val = validation_data
            X_val_flat = self._flatten(X_val)
            eval_set = [(X_train_flat, y_train), (X_val_flat, y_val)]
        
        # 3. Handle Class Weights (Dynamic Translation)
        # Keras passes a dict {0: 1.0, 1: 5.0}. XGBoost needs 'scale_pos_weight'.
        if class_weight:
            # Calculate ratio: Weight(1) / Weight(0)
            w0 = class_weight.get(0, 1.0)
            w1 = class_weight.get(1, 1.0)
            self.model.scale_pos_weight = w1 / w0
            if verbose:
                print(f"XGBoost Adapter: Applied scale_pos_weight = {self.model.scale_pos_weight:.2f}")

        # 4. Train
        self.model.fit(
            X_train_flat, y_train,
            eval_set=eval_set,
            verbose=bool(verbose)
        )
        
        # 5. Create "Fake" History Object for Plotting
        # XGBoost stores history in .evals_result()
        results = self.model.evals_result()
        
        class HistoryShim:
            def __init__(self, history_dict):
                self.history = history_dict
        
        # Map XGBoost 'validation_0'/'validation_1' to Keras 'loss'/'val_loss'
        history_map = {}
        if 'validation_0' in results:
            history_map['loss'] = results['validation_0']['logloss']
        if 'validation_1' in results:
            history_map['val_loss'] = results['validation_1']['logloss']
            
        self.history = HistoryShim(history_map)
        return self.history

    def predict(self, X, verbose=0):
        # Return probabilities (N, 1) to match Keras Sigmoid output
        X_flat = self._flatten(X)
        # predict_proba returns [prob_0, prob_1], we want column 1
        probs = self.model.predict_proba(X_flat)[:, 1] 
        return probs.reshape(-1, 1)

    def evaluate(self, X, y):
        # Mimic Keras .evaluate() -> returns [loss, accuracy]
        X_flat = self._flatten(X)
        y_pred = self.model.predict(X_flat)
        y_probs = self.model.predict_proba(X_flat)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        loss = log_loss(y, y_probs)
        return [loss, acc]

    def save(self, filepath):
        # Custom saver because .save_model uses XGBoost format
        # We rename .keras to .json to satisfy XGBoost requirements
        real_path = filepath.replace(".keras", ".json")
        self.model.save_model(real_path)
        print(f"XGBoost model saved to {real_path}")

def build_xgboost_model(input_shape):
    """
    Returns the XGBoost Adapter ready for the pipeline.
    """
    return XGBoostKerasAdapter(input_shape)



# Random Forest Model
class RandomForestKerasAdapter:
    """
    A wrapper that makes Random Forest look exactly like a Keras model.
    """
    def __init__(self, input_shape, n_estimators=100):
        self.input_shape = input_shape
        # Initialize the actual Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,            # Prevent huge trees
            min_samples_leaf=2,      # Regularization
            n_jobs=-1,               # Use all CPUs
            random_state=42
        )
        self.history = None

    def _flatten(self, X):
        """Converts 3D input (Samples, Time, Feat) to 2D"""
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def compile(self, optimizer=None, loss=None, metrics=None):
        pass

    def fit(self, X_train, y_train, epochs=None, batch_size=None, 
            validation_data=None, verbose=1, class_weight=None):
        
        # 1. Flatten Data
        X_train_flat = self._flatten(X_train)
        
        # 2. Handle Class Weights
        # Random Forest handles this natively with 'class_weight' param
        # We need to pass it during initialization or fit if we want strict parity.
        # Ideally, we set it in __init__, but we can try to set it here or rely on sample_weight.
        # For simplicity in this wrapper, we set the internal attribute if provided.
        if class_weight:
             self.model.class_weight = class_weight

        # 3. Train (Random Forest trains in one shot, not epochs)
        if verbose:
            print("Training Random Forest (this may be fast)...")
            
        self.model.fit(X_train_flat, y_train)
        
        # 4. Create Dummy History (Since RF doesn't have a loss curve)
        # We return a flat line so plot_training_loss() doesn't crash.
        class HistoryShim:
            def __init__(self):
                self.history = {'loss': [0], 'val_loss': [0]}
        
        self.history = HistoryShim()
        return self.history

    def predict(self, X, verbose=0):
        # Return probabilities (N, 1)
        X_flat = self._flatten(X)
        probs = self.model.predict_proba(X_flat)[:, 1]
        return probs.reshape(-1, 1)

    def evaluate(self, X, y):
        # Mimic Keras .evaluate()
        X_flat = self._flatten(X)
        y_pred = self.model.predict(X_flat)
        y_probs = self.model.predict_proba(X_flat)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        loss = log_loss(y, y_probs)
        return [loss, acc]

    def save(self, filepath):
        # Save as .pkl (Pickle)
        real_path = filepath.replace(".keras", ".pkl")
        with open(real_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Random Forest model saved to {real_path}")

def build_rf_model(input_shape):
    """
    Returns the Random Forest Adapter.
    """
    return RandomForestKerasAdapter(input_shape)




# SVM Model
class SVMKerasAdapter:
    """
    A wrapper that makes SVM look exactly like a Keras model.
    """
    def __init__(self, input_shape, C=1.0, kernel='rbf'):
        self.input_shape = input_shape
        # probability=True is REQUIRED for confident scores
        self.model = SVC(
            C=C, 
            kernel=kernel, 
            probability=True,  # Critical: allows .predict_proba()
            random_state=42,
            cache_size=1000,    # Speed up training
            verbose=False
        )
        self.history = None

    def _flatten(self, X):
        if len(X.shape) == 3: return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X_train, y_train, epochs=None, batch_size=None, validation_data=None, verbose=1, class_weight=None):
        X_train_flat = self._flatten(X_train)
        
        # SVM handles class weights differently (in the model init or fit)
        # We set it here if passed
        if class_weight:
            self.model.class_weight = class_weight

        if verbose: print("Training SVM (this solves a global optimization problem)...")
        self.model.fit(X_train_flat, y_train)
        
        # Dummy history
        class HistoryShim:
            def __init__(self): self.history = {'loss': [0], 'val_loss': [0]}
        self.history = HistoryShim()
        return self.history

    def predict(self, X, verbose=0):
        return self.model.predict_proba(self._flatten(X))[:, 1].reshape(-1, 1)

    def evaluate(self, X, y):
        X_flat = self._flatten(X)
        y_pred = self.model.predict(X_flat)
        y_probs = self.model.predict_proba(X_flat)[:, 1]
        return [log_loss(y, y_probs), accuracy_score(y, y_pred)]

    def save(self, filepath):
        with open(filepath.replace(".keras", ".pkl"), "wb") as f:
            pickle.dump(self.model, f)

def build_svm_model(input_shape):
    return SVMKerasAdapter(input_shape)


# KNN Model

class KNNKerasAdapter:
    """
    A wrapper that makes K-Nearest Neighbors look exactly like a Keras model.
    """
    def __init__(self, input_shape, n_neighbors=50):
        self.input_shape = input_shape
        # n_neighbors=50: Look at the 50 most similar days in history
        # weights='distance': Closer matches have more 'vote' than distant matches
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights='distance', 
            n_jobs=-1
        )
        self.history = None

    def _flatten(self, X):
        """Converts 3D input (Samples, Time, Feat) to 2D"""
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X_train, y_train, epochs=None, batch_size=None, validation_data=None, verbose=1, class_weight=None):
        X_train_flat = self._flatten(X_train)
        
        # KNN does not support class_weight natively in the same way, 
        # but 'weights="distance"' handles imbalance naturally by prioritizing close matches.
        if verbose: print(f"Training KNN (Indexing {len(X_train)} patterns)...")
        
        self.model.fit(X_train_flat, y_train)
        
        # Dummy History
        class HistoryShim:
            def __init__(self): self.history = {'loss': [0], 'val_loss': [0]}
        self.history = HistoryShim()
        return self.history

    def predict(self, X, verbose=0):
        # Return probability of Class 1 (UP)
        return self.model.predict_proba(self._flatten(X))[:, 1].reshape(-1, 1)

    def evaluate(self, X, y):
        X_flat = self._flatten(X)
        y_pred = self.model.predict(X_flat)
        y_probs = self.model.predict_proba(X_flat)[:, 1]
        return [log_loss(y, y_probs), accuracy_score(y, y_pred)]

    def save(self, filepath):
        # Save as .pkl
        real_path = filepath.replace(".keras", ".pkl")
        with open(real_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"KNN model saved to {real_path}")

def build_knn_model(input_shape):
    """
    Returns the KNN Adapter.
    """
    return KNNKerasAdapter(input_shape)