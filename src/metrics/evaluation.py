import numpy as np 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import learning_curve
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, log_loss
)

warnings.filterwarnings('ignore')

def check_metrics(model, name, X_train, y_train, X_test, y_test):
    """ check accuracy, loss, precision, recall, f1-score, tpr, fpr, confusion matrix and precision-recall curve for test data """
    def get_predictions(model, X, y):
        """ get the predictions for the given model one data X & y """
        true_multi = y['Attack Type'].values
        true_binary = y['status'].values

        # get predictions
        pred = model.predict(X)
        pred_multi = pred[:, 0] # multi-class predictions
        pred_binary = pred[:, 1] # binary predictions

        # get probabilities if the model supports
        prob_multi = None
        prob_binary = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            prob_multi = proba[0] # multi-class probabilities
            prob_binary = proba[1][:, 1] # binary probabilities

        return pred_multi, pred_binary, true_multi, true_binary, prob_multi, prob_binary

    ## ========================================================================================================= ##
    ## title
    print("="*100)
    print(" "*40 + "\033[1;34m" + name.upper() + " REPORT" + "\033[0m" + " "*10)
    print("="*100)

    ## ========================================================================================================= ##
    # get predictions
    train_pred_multi, train_pred_binary, train_true_multi, train_true_binary, train_prob_multi, train_prob_binary = get_predictions(model, X_train, y_train)
    test_pred_multi, test_pred_binary, test_true_multi, test_true_binary, test_prob_multi, test_prob_binary = get_predictions(model, X_test, y_test)

    # calculate accuracies
    train_acc_multi = accuracy_score(train_true_multi, train_pred_multi)
    train_acc_binary = accuracy_score(train_true_binary, train_pred_binary)
    train_acc_avg = (train_acc_binary + train_acc_multi) / 2
    test_acc_multi = accuracy_score(test_true_multi, test_pred_multi)
    test_acc_binary = accuracy_score(test_true_binary, test_pred_binary)
    test_acc_avg = (test_acc_binary + test_acc_multi) / 2

    # calculate losses
    train_loss_multi = log_loss(train_true_multi, train_prob_multi) if train_prob_multi is not None else float('nan')
    train_loss_binary = log_loss(train_true_binary, train_prob_binary) if train_prob_binary is not None else float('nan')
    train_loss_avg = (train_loss_multi + train_loss_binary) / 2
    test_loss_multi = log_loss(test_true_multi, test_prob_multi) if test_prob_multi is not None else float('nan')
    test_loss_binary = log_loss(test_true_binary, test_prob_binary) if test_prob_binary is not None else float('nan')
    test_loss_avg = (train_loss_multi + train_loss_binary) / 2

    ## ========================================================================================================= ##
    # Print accuracy report
    print("="*100)
    print(f"\n{'':<35} < === Accuracy & Loss Report === >")
    print()
    print(f"{'':<20}   Multi-class  |   Binary   |   Average")
    print(f"Training Accuracy:   {train_acc_multi:10.4f}     |   {train_acc_binary:6.4f}   |   {train_acc_avg:6.4f}")
    print(f"Test Accuracy:       {test_acc_multi:10.4f}     |   {test_acc_binary:6.4f}   |   {test_acc_avg:6.4f}")
    print("                            -       |      -     |      -")
    print(f"Training Loss:       {train_loss_multi:10.4f}     |   {train_loss_binary:6.4f}   |   {train_loss_avg:6.4f}")
    print(f"Test Loss:           {test_loss_multi:10.4f}     |   {test_loss_binary:6.4f}   |   {test_loss_avg:6.4f}")
    print()

    ## ========================================================================================================= ##
    # Binary Classification | test set metrics
    # calculate tpr & fpr for binary classification
    tn, fp, fn, tp = confusion_matrix(test_true_binary, test_pred_binary).ravel()
    tpr_binary = (tp / (tp + fn)).item()
    fpr_binary = (fp / (fp + tn)).item()

    precision_binary = precision_score(test_true_binary, test_pred_binary)
    recall_binary = recall_score(test_true_binary, test_pred_binary)
    f1_binary = f1_score(test_true_binary, test_pred_binary)

    print("="*100)
    print(f"\n{'':<32} < === Binary Classification Report === >")
    print()
    print(f"Precision: {precision_binary:.4f}")
    print(f"Recall: {recall_binary:.4f}")
    print(f"F1 Score: {f1_binary:.4f}")

    print(f"True Positive Rate: {tpr_binary:.4f}")
    print(f"False Positive Rate: {fpr_binary:.4f}")
    print()

    # plot confusion matrix
    cm_binary = confusion_matrix(test_true_binary, test_pred_binary)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues')
    plt.title('Binary Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print()

    # binary precision-recall curve
    if test_prob_binary is not None:
        precision, recall, _ = precision_recall_curve(test_true_binary, test_prob_binary)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker='.')
        plt.title('Binary Classification Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.show()
    else:
        print("Precision-Recall curve for binary classification skipped (model does not support predict_proba)")
    print()

    ## ========================================================================================================= ##
    # Multi-class Classification | test set metrics
    # calculate tpr & fpr for binary classification
    tpr_list = []
    fpr_list = []
    output_class = len(np.unique(test_true_multi))
    for i in range(output_class):
        # convert multiclass labels to binary
        y_true_binary = (test_true_multi == i).astype(int)
        y_pred_binary = (test_pred_multi == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_multi = np.mean(tpr_list).item()
    fpr_multi = np.mean(fpr_list).item()
    precision_multi = precision_score(test_true_multi, test_pred_multi, average='weighted')
    recall_multi = recall_score(test_true_multi, test_pred_multi, average='weighted')
    f1_multi = f1_score(test_true_multi, test_pred_multi, average='weighted')

    print("="*100)
    print(f"\n{'':<26} < === Multi-class Classification Report === >")
    print()
    print(f"Precision: {precision_multi:.4f}")
    print(f"Recall: {recall_multi:.4f}")
    print(f"F1 Score: {f1_multi:.4f}")
    print(f"True Positive Rate: {tpr_multi:.4f}")
    print(f"False Positive Rate: {fpr_multi:.4f}")
    print()

    # plot confusion matrix
    cm_multi = confusion_matrix(test_true_multi, test_pred_multi)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues')
    plt.title('Multi-class Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print()

    # multi-class precision-recall curve
    if test_prob_multi is not None:
        plt.figure(figsize=(8, 5))
        for i in range(output_class):
            # Binarize true labels for class i
            true_binary = (test_true_multi == i).astype(int)
            prob_class = test_prob_multi[:, i]
            precision, recall, _ = precision_recall_curve(true_binary, prob_class)
            plt.plot(recall, precision, label=f"Class {i}")
        plt.title('Multi-class Classification Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(title='class', loc='best')
        plt.grid(True)
        plt.show()
    else:
        print("Precision-Recall curve for binary classification skipped (model does not support predict_proba)")
    print()

    ## ========================================================================================================= ##
    # Average Results | test set metrics


    tpr_avg = ((tpr_binary + tpr_multi) / 2)
    fpr_avg = ((fpr_binary + fpr_multi) / 2)

    precision_avg = (precision_binary + precision_multi) / 2
    recall_avg = (recall_binary + recall_multi) / 2
    f1_avg = (f1_binary + f1_multi) / 2

    print("="*100)
    print(f"\n{'':<30} < === Average Classification Report === >")
    print()
    print(f"Precision: {precision_avg:.4f}")
    print(f"Recall: {recall_avg:.4f}")
    print(f"F1 Score: {f1_avg:.4f}")

    print(f"True Positive Rate: {tpr_avg:.4f}")
    print(f"False Positive Rate: {fpr_avg:.4f}")
    print("="*100)

    # return 3 tuples -> binary_classification_report, multi_classification_report, avg_classification_report
    return ((train_acc_binary, test_acc_binary, train_loss_binary, test_loss_binary, f1_binary, precision_binary, recall_binary, tpr_binary, fpr_binary),
            (train_acc_multi, test_acc_multi, train_loss_multi, test_loss_multi, f1_multi, precision_multi, recall_multi, tpr_multi, fpr_multi),
            (train_acc_avg, test_acc_avg, train_loss_avg, test_loss_avg, f1_avg, precision_avg, recall_avg, tpr_avg, fpr_avg))


def evaluate_dnn_result(model, name, train_loader, test_loader, output_class, device):
    """ evaluate the deep neural network model on training, and test set for mlp and cnn models """
    """ print the training, and test set accuracy first """
    """ print the classification report (accuracy, precision, recall, f1-score, tpr, fpr), confusion matrix
        and precision-recall curve for multi-class and binary classification """

    def get_prediction(loader):
        """ get the predicitons for the model """
        total_pred_multi = []
        total_pred_binary = []
        total_true_multi = []
        total_true_binary = []
        total_prob_multi = []
        total_prob_binary = []

        model.eval()  # enable evaluation mode
        with torch.no_grad():
            for batch_X, batch_y_multi, batch_y_binary in loader:
                batch_X, batch_y_multi, batch_y_binary =  batch_X.to(device), batch_y_multi.to(device), batch_y_binary.to(device)

                # forward pass
                pred_multi, pred_binary = model(batch_X)

                # prediction
                _, predicted_multi = torch.max(pred_multi, 1) # modified remove .data
                predicted_binary = (pred_binary > 0.5).float()

                # move to cpu and convert to numpy
                total_pred_multi.extend(predicted_multi.cpu().numpy())
                total_pred_binary.extend(predicted_binary.cpu().numpy().flatten())
                total_true_multi.extend(batch_y_multi.cpu().numpy())
                total_true_binary.extend(batch_y_binary.cpu().numpy().flatten())
                total_prob_multi.extend(torch.softmax(pred_multi, dim=1).cpu().numpy())
                total_prob_binary.extend(pred_binary.cpu().numpy().flatten())

        return np.array(total_pred_multi), np.array(total_pred_binary), np.array(total_true_multi), np.array(total_true_binary), np.array(total_prob_multi), np.array(total_prob_binary)


    ## ========================================================================================================= ##
    ## title
    print("="*100)
    print(" "*40 + "\033[1;34m" + name.upper() + " REPORT" + "\033[0m" + " "*10)
    print("="*100)

    ## ========================================================================================================= ##
    # get predictions
    train_pred_multi, train_pred_binary, train_true_multi, train_true_binary, train_prob_multi, train_prob_binary = get_prediction(train_loader)
    test_pred_multi, test_pred_binary, test_true_multi, test_true_binary, test_prob_multi, test_prob_binary = get_prediction(test_loader)

    # calculate accuracies
    train_acc_multi = accuracy_score(train_true_multi, train_pred_multi)
    train_acc_binary = accuracy_score(train_true_binary, train_pred_binary)
    train_acc_avg = (train_acc_binary + train_acc_multi) / 2
    test_acc_multi = accuracy_score(test_true_multi, test_pred_multi)
    test_acc_binary = accuracy_score(test_true_binary, test_pred_binary)
    test_acc_avg = (test_acc_binary + test_acc_multi) / 2

    # calculate losses
    train_loss_multi = log_loss(train_true_multi, train_prob_multi) if train_prob_multi is not None else float('nan')
    train_loss_binary = log_loss(train_true_binary, train_prob_binary) if train_prob_binary is not None else float('nan')
    train_loss_avg = (train_loss_multi + train_loss_binary) / 2
    test_loss_multi = log_loss(test_true_multi, test_prob_multi) if test_prob_multi is not None else float('nan')
    test_loss_binary = log_loss(test_true_binary, test_prob_binary) if test_prob_binary is not None else float('nan')
    test_loss_avg = (train_loss_multi + train_loss_binary) / 2

    ## ========================================================================================================= ##
    # Print accuracy report
    print("="*100)
    print(f"\n{'':<35} < === Accuracy & Loss Report === >")
    print()
    print(f"{'':<20}   Multi-class  |   Binary   |   Average")
    print(f"Training Accuracy:   {train_acc_multi:10.4f}     |   {train_acc_binary:6.4f}   |   {train_acc_avg:6.4f}")
    print(f"Test Accuracy:       {test_acc_multi:10.4f}     |   {test_acc_binary:6.4f}   |   {test_acc_avg:6.4f}")
    print("                            -       |      -     |      -")
    print(f"Training Loss:       {train_loss_multi:10.4f}     |   {train_loss_binary:6.4f}   |   {train_loss_avg:6.4f}")
    print(f"Test Loss:           {test_loss_multi:10.4f}     |   {test_loss_binary:6.4f}   |   {test_loss_avg:6.4f}")
    print()

    ## ========================================================================================================= ##
    # Binary Classification | test set metrics
    # calculate tpr & fpr for binary classification
    tn, fp, fn, tp = confusion_matrix(test_true_binary, test_pred_binary).ravel()
    tpr_binary = (tp / (tp + fn)).item()
    fpr_binary = (fp / (fp + tn)).item()

    precision_binary = precision_score(test_true_binary, test_pred_binary)
    recall_binary = recall_score(test_true_binary, test_pred_binary)
    f1_binary = f1_score(test_true_binary, test_pred_binary)

    print("="*100)
    print(f"\n{'':<32} < === Binary Classification Report === >")
    print()
    print(f"Precision: {precision_binary:.4f}")
    print(f"Recall: {recall_binary:.4f}")
    print(f"F1 Score: {f1_binary:.4f}")

    print(f"True Positive Rate: {tpr_binary:.4f}")
    print(f"False Positive Rate: {fpr_binary:.4f}")
    print()

    # plot confusion matrix
    cm_binary = confusion_matrix(test_true_binary, test_pred_binary)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues')
    plt.title('Binary Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print()

    # binary precision-recall curve
    if test_prob_binary is not None:
        precision, recall, _ = precision_recall_curve(test_true_binary, test_prob_binary)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker='.')
        plt.title('Binary Classification Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.show()
    else:
        print("Precision-Recall curve for binary classification skipped (model does not support predict_proba)")
    print()

    ## ========================================================================================================= ##
    # Multi-class Classification | test set metrics
    # calculate tpr & fpr for binary classification
    tpr_list = []
    fpr_list = []
    output_class = len(np.unique(test_true_multi))
    for i in range(output_class):
        # convert multiclass labels to binary
        y_true_binary = (test_true_multi == i).astype(int)
        y_pred_binary = (test_pred_multi == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_multi = np.mean(tpr_list).item()
    fpr_multi = np.mean(fpr_list).item()
    precision_multi = precision_score(test_true_multi, test_pred_multi, average='weighted')
    recall_multi = recall_score(test_true_multi, test_pred_multi, average='weighted')
    f1_multi = f1_score(test_true_multi, test_pred_multi, average='weighted')

    print("="*100)
    print(f"\n{'':<26} < === Multi-class Classification Report === >")
    print()
    print(f"Precision: {precision_multi:.4f}")
    print(f"Recall: {recall_multi:.4f}")
    print(f"F1 Score: {f1_multi:.4f}")
    print(f"True Positive Rate: {tpr_multi:.4f}")
    print(f"False Positive Rate: {fpr_multi:.4f}")
    print()

    # plot confusion matrix
    cm_multi = confusion_matrix(test_true_multi, test_pred_multi)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues')
    plt.title('Multi-class Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print()

    # multi-class precision-recall curve
    if test_prob_multi is not None:
        plt.figure(figsize=(8, 5))
        for i in range(output_class):
            # Binarize true labels for class i
            true_binary = (test_true_multi == i).astype(int)
            prob_class = test_prob_multi[:, i]
            precision, recall, _ = precision_recall_curve(true_binary, prob_class)
            plt.plot(recall, precision, label=f"Class {i}")
        plt.title('Multi-class Classification Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(title='class', loc='best')
        plt.grid(True)
        plt.show()
    else:
        print("Precision-Recall curve for binary classification skipped (model does not support predict_proba)")
    print()

    ## ========================================================================================================= ##
    # Average Results | test set metrics


    tpr_avg = ((tpr_binary + tpr_multi) / 2)
    fpr_avg = ((fpr_binary + fpr_multi) / 2)

    precision_avg = (precision_binary + precision_multi) / 2
    recall_avg = (recall_binary + recall_multi) / 2
    f1_avg = (f1_binary + f1_multi) / 2

    print("="*100)
    print(f"\n{'':<30} < === Average Classification Report === >")
    print()
    print(f"Precision: {precision_avg:.4f}")
    print(f"Recall: {recall_avg:.4f}")
    print(f"F1 Score: {f1_avg:.4f}")

    print(f"True Positive Rate: {tpr_avg:.4f}")
    print(f"False Positive Rate: {fpr_avg:.4f}")
    print("="*100)

    # return 3 tuples -> binary_classification_report, multi_classification_report, avg_classification_report
    return ((train_acc_binary, test_acc_binary, train_loss_binary, test_loss_binary, f1_binary, precision_binary, recall_binary, tpr_binary, fpr_binary),
            (train_acc_multi, test_acc_multi, train_loss_multi, test_loss_multi, f1_multi, precision_multi, recall_multi, tpr_multi, fpr_multi),
            (train_acc_avg, test_acc_avg, train_loss_avg, test_loss_avg, f1_avg, precision_avg, recall_avg, tpr_avg, fpr_avg))


def plot_learning_curve(model, X, y, name, version):
    """ get the model & dataset calculate the training and cross validation losses and plot the learning curve """
    def custom_neg_log_loss(model, X, y):
        """ custom loss function to calculate the negative log loss """
        y_pred_proba = model.predict_proba(X) # get probabilities
        total_loss = 0
        for i, y_label in enumerate(y.columns):
            y_pred = np.clip(y_pred_proba[i], 1e-15, 1 - 1e-15) # get the probabilities of both labels per iterations
            # calculate loss of both label per iterations
            loss = log_loss(y[y_label], y_pred)
            total_loss += loss
        return -total_loss / len(y.columns) # return the average loss

    # get loss values
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring= custom_neg_log_loss,
        train_sizes = np.linspace(0.1, 1.0, 50)
    )

    train_losses = -train_scores.mean(axis=1)
    val_losses = -val_scores.mean(axis=1)
    #train_std = train_scores.std(axis=1)
    #val_std = val_scores.std(axis=1)

    # debuging statements
    #print("Train size:", train_sizes)
    print("Train loss:", round(train_losses[-1], 4))
    print("Val loss:", round(val_losses[-1], 4))

    # ploting the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_losses, label='Training Loss')
    plt.plot(train_sizes, val_losses, label='Validation Loss (CV)')
    #plt.fill_between(train_sizes, train_losses - train_std, train_losses + train_std, alpha=0.1)
    #plt.fill_between(train_sizes, val_losses - val_std, val_losses + val_std, alpha=0.1)
    plt.title(f"{name} Learning Curve ({version})")
    plt.xlabel("Training Set Size")
    plt.ylabel("Losses")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_curves(history, name, version):
    """ get the model history data as dictionary and plot the learning curve (loss & accuracy curves) """ 
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title(f"{name} Learning Curve ({version})")
    plt.xlabel("Epochs")
    plt.ylabel('Losses')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print()
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc_multi'], label='Train Multi-class Acc')
    plt.plot(history['val_acc_multi'], label='Val Multi-class Acc')
    plt.plot(history['train_acc_binary'], label='Train Binary Acc')
    plt.plot(history['val_acc_binary'], label='Val Binary Acc')
    plt.plot(history['train_acc_avg'], label='Train Avg Acc')
    plt.plot(history['val_acc_avg'], label='Val Avg Acc')
    plt.title(f"{name} Accuracy Curve ({version})")
    plt.xlabel("Epochs")
    plt.ylabel('Losses')
    plt.legend()
    plt.tight_layout()
    plt.show()



def test_model(model, test_loader, device):
    """ evaluates the trained model on the test dataset for both multi-class and binary classification task. """
    model.eval()  # enable evaluation mode
    correct = 0
    correct_binary = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y_multi, batch_y_binary in test_loader:
            batch_X, batch_y_multi, batch_y_binary =  batch_X.to(device), batch_y_multi.to(device), batch_y_binary.to(device)

            # forward pass
            pred_multi, pred_binary = model(batch_X)

            # predictions
            _, predicted_multi = torch.max(pred_multi, 1) # modified remove .data
            predicted_binary = (pred_binary > 0.5).float()

            # calculate accuracy
            correct += (predicted_multi == batch_y_multi).sum().item()
            correct_binary += (predicted_binary == batch_y_binary).sum().item()
            total += batch_y_multi.size(0)

    print(f"Test Multi-class Accuracy: {correct/total:.4f}")
    print(f"Test Binary Accuracy: {correct_binary/total:.4f}")
    print(f"Test Average Accuracy: {((correct/total) + (correct_binary/total)) / 2}")


def get_model_params(model):
    """ returns the number of trainable parameters in the model """
    print(f"Total Number of Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")