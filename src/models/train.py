from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
import torch 
import copy

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

def train_epoch(model, data_loader, criterion_multi, criterion_binary, optimizer=None, is_training=True, device='cuda'):
    """ this function contain the model training evaluation epoch loop to reduce code redundant """
    total_loss = 0
    correct_multi = 0
    correct_binary = 0
    total_samples = 0

    # Set model mode (train or eval)
    model.train() if is_training else model.eval()

    for batch_X, batch_y_multi, batch_y_binary in data_loader:
        batch_X, batch_y_multi, batch_y_binary = batch_X.to(device), batch_y_multi.to(device), batch_y_binary.to(device)

        # forward pass
        pred_multi, pred_binary = model(batch_X)

        # calculate loss
        loss_multi = criterion_multi(pred_multi, batch_y_multi)
        loss_binary = criterion_binary(pred_binary, batch_y_binary)
        loss = loss_multi + loss_binary

        if is_training:
            # bacward pass
            optimizer.zero_grad()
            loss.backward()
            # update grad
            optimizer.step()

        # calculate total epoch loss and accuracy
        total_loss += loss.item()
        _, predicted_multi = torch.max(pred_multi, 1)
        predicted_binary = (pred_binary > 0.5).float()

        correct_multi += (predicted_multi == batch_y_multi).sum().item()
        correct_binary += (predicted_binary == batch_y_binary).sum().item()
        total_samples += batch_y_multi.size(0)

    avg_loss = total_loss / len(data_loader)
    acc_multi = correct_multi / total_samples
    acc_binary = correct_binary / total_samples
    acc_avg = (acc_multi + acc_binary) / 2

    return avg_loss, acc_multi, acc_binary, acc_avg

def train_model(model, train_loader, val_loader, criterion_multi, criterion_binary, optimizer, scheduler, num_epochs, device, patience=5, tune=False):
    # total training losses & accuracy
    train_losses = []
    train_acc_multi = []
    train_acc_binary = []
    train_acc_avg = []

    # total validation losses and accuracy
    val_losses = []
    val_acc_multi = []
    val_acc_binary = []
    val_acc_avg = []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # trainig loop
    for epoch in range(num_epochs):
        avg_train_loss, acc_train_multi, acc_train_binary, acc_train_avg = train_epoch(
            model,
            train_loader,
            criterion_multi,
            criterion_binary,
            optimizer=optimizer,
            is_training=True,
            device=device
        )

        # store training metrics
        train_losses.append(avg_train_loss)
        train_acc_multi.append(acc_train_multi)
        train_acc_binary.append(acc_train_binary)
        train_acc_avg.append(acc_train_avg)

        # validation phase
        avg_val_loss, acc_val_multi, acc_val_binary, acc_val_avg = train_epoch(
            model,
            val_loader,
            criterion_multi,
            criterion_binary,
            optimizer=None,
            is_training=False,
            device=device
        )

        # store validation metrics
        val_losses.append(avg_val_loss)
        val_acc_multi.append(acc_val_multi)
        val_acc_binary.append(acc_val_binary)
        val_acc_avg.append(acc_val_avg)

        if tune==False:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc Avg: {train_acc_avg[-1]:.4f}, Val Acc Avg: {val_acc_avg[-1]:.4f}, Train Acc Multi: {train_acc_multi[-1]:.4f}, Train Acc Binary: {train_acc_binary[-1]:.4f}, Val Acc Multi: {val_acc_multi[-1]:.4f}, Val Acc Binary: {val_acc_binary[-1]:.4f}")

        # step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # saving the best model weights
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epoch > 10 and epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    model.load_state_dict(best_model_wts)   # loading the best model weights

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acc_multi': train_acc_multi,
        'train_acc_binary': train_acc_binary,
        'train_acc_avg': train_acc_avg,
        'val_acc_multi': val_acc_multi,
        'val_acc_binary': val_acc_binary,
        'val_acc_avg': val_acc_avg
    }
    return model, history

def train_final_model(model, train_loader, criterion_multi, criterion_binary, optimizer, num_epochs, device):
    """ train the final model on the training dataset only """
    train_losses = []
    train_acc_multi = []
    train_acc_binary = []
    train_acc_avg = []

    # Training loop (no validation)
    for epoch in range(num_epochs):
        avg_train_loss, acc_train_multi, acc_train_binary, acc_train_avg = train_epoch(
            model,
            train_loader,
            criterion_multi,
            criterion_binary,
            optimizer=optimizer,
            is_training=True,
            device=device
        )

        train_losses.append(avg_train_loss)
        train_acc_multi.append(acc_train_multi)
        train_acc_binary.append(acc_train_binary)
        train_acc_avg.append(acc_train_avg)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc Avg: {acc_train_avg:.4f}, Train Acc Multi: {acc_train_multi:.4f}, Train Acc Binary: {acc_train_binary:.4f}")

    history = {
        'train_losses': train_losses,
        'train_acc_multi': train_acc_multi,
        'train_acc_binary': train_acc_binary,
        'train_acc_avg': train_acc_avg
    }

    return model, history
