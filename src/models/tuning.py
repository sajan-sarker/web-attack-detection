from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import time
import torch
import torch.nn as nn 
import torch.optim as optim

from optuna.trial import Trial
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def measure_time(sec, name, Tune=False):
    """ Measure the time taken for training the model """
    hours, rem = divmod(sec, 3600)
    minutes, seconds = divmod(rem, 60)
    if Tune:
        print(f"{name} Hyperparameter Tuning Time: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")
    else:
        print(f"{name} Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s \n")



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

def objective(trial, train_dataset, val_dataset, input_dim, output_dim, device):
    # suggested parameters
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 10)
    neurons_per_layers = trial.suggest_int('neurons_per_layers', 32, 512, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5, step=0.05)
    num_epochs = trial.suggest_int('num_epochs', 20, 100, step=20)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    #learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00005])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'SGD', 'RMSprop'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # init model
    model = MultiLayerPerceptron(
        input_dim = input_dim,
        output_dim = output_dim,
        num_hidden_layers = num_hidden_layers,
        neurons_per_layers = neurons_per_layers,
        dropout_rate = dropout_rate
    ).to(device)

    # define loss functions and optimizer with weight decay
    criterion_multi = nn.CrossEntropyLoss()
    criterion_binary = nn.BCELoss()

    # select optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # train the model
    _, history = train_model(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion_multi = criterion_multi,
        criterion_binary = criterion_binary,
        optimizer = optimizer,
        scheduler = scheduler,
        num_epochs = num_epochs,
        device = device,
        tune=True
    )

    # return the best validation loss as the objective to minimize
    best_val_loss = min(history['val_losses'])
    return best_val_loss

def tune_hyperparameters(train_dataset, val_dataset, input_dim, output_dim, device, n_trials=50):
    """ perform hyperparameter tuning using optuna """
    # create optuna study with minimize validation loss
    study = optuna.create_study(direction='minimize')
    start = time.time()
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, input_dim, output_dim, device),
        n_trials=n_trials,
        n_jobs=3,
        show_progress_bar=True
    )
    measure_time((time.time()-start), "MLP", True)
    print("Number of finished trials:", len(study.trials))
    print("Best trial: ")
    trial = study.best_trial

    print(f"Validation Loss: {trial.value:4f}")
    print("Best Parameters: ")
    for key, value in trial.params.items():
        print(f" {key}: {value}")

    return study.best_params