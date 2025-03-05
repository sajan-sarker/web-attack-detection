from sikitlearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sikitlearn.metrics import classification_report, roc_curve, auc, display_precision_recall_curve, display_roc_curve, disp
import matplotlib.pyplot as plt
import seaborn as sns

def check_metrics(model, name, X_train, y_train, X_val, y_val, X_test, y_test):
    """ Function to check the metrics of the model """
    """ It'll print the training, validation and test accuracy of the model 
        It'll also print the classification report for multi-class and binary classification
        It'll also print the confusion matrix for multi-class and binary classification
    """
    train = model.predict(X_train)
    val = model.predict(X_val)
    test = model.predict(X_test)
    
    train_acc1 = accuracy_score(y_train['Attack Type'], train[:,0])
    train_acc2 = accuracy_score(y_train['status'], train[:,1])
    val_acc1 = accuracy_score(y_val['Attack Type'], val[:,0])
    val_acc2 = accuracy_score(y_val['status'], val[:,1])
    test_acc1 = accuracy_score(y_test['Attack Type'], test[:,0])
    test_acc2 = accuracy_score(y_test['status'], test[:,1])
    
    print(f"{name} Train Accuracy- Attack Type: {train_acc1:.4}, Status: {train_acc2:.4}, Average: {((train_acc1+train_acc2)/2):.4}")
    print(f"{name} Validation Accuracy- Attack Type: {val_acc1:.4}, Status: {val_acc2:.4}, Average: {((val_acc1+val_acc2)/2):.4}")
    print(f"{name} Test Accuracy- Attack Type: {test_acc1:.4}, Status: {test_acc2:.4}, Average: {((test_acc1+test_acc2)/2):.4}")
    
    # multi-class classification report
    print("\nClassification Report for multi-class classification")
    print(classification_report(y_test['Attack Type'], test[:,0]))
    
    # binary classification report
    print("\nClassification Report for binary classification")
    print(classification_report(y_test['status'], test[:,1]))
    
    attack_cm = confusion_matrix(y_test['Attack Type'], test[:, 0])
    label_cm = confusion_matrix(y_test['status'], test[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    sns.heatmap(attack_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Multi-class Classification Confusion Matrix')
    axes[0].set_xlabel('Predicted Result')
    axes[0].set_ylabel('Actual Result')

    sns.heatmap(label_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Binary Classification Confusion Matrix')
    axes[1].set_xlabel('Predicted Result')
    axes[1].set_ylabel('Actual Result')

    plt.tight_layout()
    plt.show()