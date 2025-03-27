from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

def perform_hyperparameter_tuning(model, parameters, X_train, y_train):
    """ Perform hyperparameter tuning on the model """
    """ Return the best model and best parameters """
    model = MultiOutputClassifier(model)
    grid_search = GridSearchCV(
        estimator= model,
        param_grid=parameters,
        cv=5,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print("\nBest Parameters:", best_params)
    return best_params