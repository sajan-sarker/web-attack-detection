import numpy as np
from sklearn.metrics import log_loss

class EnsembbleModel:
    def __init__(self, models):
        self.models = models
        self.model_predictions = {}
        self.weights = {}
        self.n_classes_multi = None

    def multi_log_loss(self, y, y_probas):
        loss1 = log_loss(y.iloc[:, 0], y_probas[0], normalize=True)
        loss2 = log_loss(y.iloc[:, 1], y_probas[1], normalize=True)
        return (loss1 + loss2) / 2

    def fit(self, X, y):
        self.X = X
        self.n_classes_multi = len(np.unique(y.iloc[:,0]))
        for name, model in self.models.items():
            probas = model.predict_proba(X)
            self.model_predictions[name] = probas

            ll = self.multi_log_loss(y, probas)
            self.weights[name] = 1 / ll if ll > 0 else 1.0  # inverse log loss as weight
            #print(f"{name} Loss: {ll:.4f}")    # tracing

        total_weight = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total_weight
            #print(f"{name} Weight: {self.weights[name]:.4f}")      # tracing
        #print(f'total: {sum(self.weights.values())}')      # tracing

    def predict_proba(self, X):
        n_samples = len(X)
        n_classes_multi = self.n_classes_multi

        # init probability arrays for both outputs
        proba_multi = np.zeros((n_samples, n_classes_multi))
        proba_binary = np.zeros((n_samples, 2))

        # for name in self.models:
        #     multi_pred, binary_pred = self.model_predictions[name]
        for name, model in self.models.items():
            multi_pred, binary_pred = model.predict_proba(X)

            # Apply weights to each output separately
            proba_multi += self.weights[name] * multi_pred
            proba_binary += self.weights[name] * binary_pred

        return proba_multi, proba_binary

    def predict(self, X):
        proba_multi, proba_binary = self.predict_proba(X)

        # Get final predictions for both outputs
        pred_multi = np.argmax(proba_multi, axis=1)
        pred_binary = np.argmax(proba_binary, axis=1)

        # Combine predictions into final output
        return np.column_stack((pred_multi, pred_binary))

    def get_weights(self):
        for name in self.weights:
            print(f"{name} Weight: {self.weights[name]:.4f}")
        return self.weights

    def get_models(self):
        return self.models