import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from IPython.display import display

class XAIAnalyzer:
    def __init__(self, model):
        """ Initialize the XAIAnalyzer with the model."""
        self.model = model
        self.feature_names = []
        self.classes = []
        self.scaler = None
        self.X_test = None
        self.n_class = 0

    def fit(self, features, decoder, scaler, X_test):
        """ Fit the XAIAnalyzer with the provided features, decoder, scaler, and test data."""
        self.feature_names = features
        self.classes = list(decoder.keys())
        self.scaler = scaler
        self.X_test = X_test
        self.n_class = len(self.classes)

    def predict_proba(self,X):
        return self.model.predict_proba(X)[0]
    
    def analyze_lime(self, data, sample_idx: int = 0) -> None:
        """ Generate LIME explanations using scaled values for predictions and original values for display """
        value_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        explainer = LimeTabularExplainer(
            training_data=self.X_test.values,
            feature_names=self.feature_names,
            class_names=self.classes,
            mode='classification',
            random_state=42
        )
        
        # explain new value
        sample_df = value_df.iloc[0]   # As DataFrame for feature names
        sample= data[0]   # as array for lime
        exp = explainer.explain_instance(
            data_row=sample,
            predict_fn=self.predict_proba,
            num_features=len(self.feature_names),
            top_labels=self.n_class
        )
        
        # get predicted class and probabilities
        probs = self.predict_proba(data)[0]
        probs = np.array(probs).flatten()

        pred_class = np.argmax(probs)
        print(f"Predicted class: {self.classes[pred_class]}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})
        
        #print(probs)    #debug 
        # Left: Prediction probabilities
        ax1.barh(range(self.n_class), probs, color='skyblue', edgecolor='black')
        ax1.set_yticks(range(self.n_class))
        ax1.set_yticklabels(self.classes)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Prediction probabilities')
        for i, prob in enumerate(probs):
            ax1.text(prob + 0.02, i, f'{prob:.2f}', va='center')
            
        # Right: Feature contributions for the predicted class
        exp_list = exp.as_list(label=pred_class)
        features, weights = zip(*exp_list)
        colors = ['orange' if w > 0 else 'blue' for w in weights]
        ax2.barh(features, weights, color=colors)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features)
        ax2.set_ylim(-0.5, len(features) - 0.5)
        ax2.set_xlim(-max(abs(np.array(weights))) * 1.2, max(abs(np.array(weights))) * 1.2)
        for i, w in enumerate(weights):
            ax2.text(w + (0.02 if w > 0 else -0.02), i, f'{w:.2f}', va='center', ha='left' if w > 0 else 'right')
        ax2.set_title(f"Class {pred_class}", color='orange')

        plt.tight_layout()
        plt.title(f"Predicted Class: {self.classes[pred_class]}")
        plt.show()