from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

def train_base_model(model, name, X_train, y_train, X_test, y_test):
    """ Train the model and print the training and test accuracy """
    """ Print the training and test accuracy of the model """
    model = MultiOutputClassifier(model)
    model.fit(X_train, y_train)

    train = model.predict(X_train)
    test = model.predict(X_test)

    train_multi = accuracy_score(y_train['Attack Type'], train[:,0])
    train_binary = accuracy_score(y_train['status'], train[:,1])
    test_multi = accuracy_score(y_test['Attack Type'], test[:,0])
    test_acc2 = accuracy_score(y_test['status'], test[:,1])

    print(f"{name} =====================================================")
    print(f"Train Accuracy- Multi-class: {train_multi:.4}, Binary: {train_binary:.4}, Average: {((train_multi+train_binary)/2):.4}")
    print(f"Test Accuracy- Multi-class: {test_multi:.4}, Binary: {test_acc2:.4}, Average: {((test_multi+test_acc2)/2):.4}")

