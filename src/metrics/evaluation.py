import numpy as np 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

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