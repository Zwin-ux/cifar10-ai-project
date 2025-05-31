import torch
import torch.nn as nn
import torch.optim as optim
import time # For potential timing/logging later

class CVTrainer:
    def __init__(self, model, trainloader, valloader, testloader, criterion, optimizer, scheduler, device, epochs, patience=7, min_delta=0.001, model_save_path='best_model.pth'):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.best_model_state = None

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in self.trainloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = val_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self):
        print(f"Starting training for {self.epochs} epochs on {self.device}...")
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()

            val_loss, val_acc = float('inf'), 0.0 # Default if no valloader
            if self.valloader:
                val_loss, val_acc = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            if self.valloader:
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping logic
            if self.valloader: # Only apply early stopping if there's a validation set
                if val_acc > self.best_val_acc + self.min_delta:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                    self.best_model_state = self.model.state_dict()
                    torch.save(self.best_model_state, self.model_save_path)
                    print(f"Validation accuracy improved. Saved model to {self.model_save_path}")
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            elif train_acc > self.best_val_acc + self.min_delta : # If no valloader, use train_acc for saving best model (no early stopping)
                self.best_val_acc = train_acc # Using best_val_acc to store best_train_acc in this case
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.model_save_path)
                print(f"Best training accuracy so far. Saved model to {self.model_save_path}")


        print('Finished Training.')
        if self.valloader:
            print(f'Best Validation Accuracy: {self.best_val_acc*100:.2f}%')
        elif self.best_model_state: # if no valloader, but we saved a model
             print(f'Best Training Accuracy: {self.best_val_acc*100:.2f}%')


        # Load best model for testing if early stopping occurred or if saving based on train acc
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.train_losses, self.train_accs, self.val_losses, self.val_accs


    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        print(f"Testing model on {self.device}...")
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = test_loss / total
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy*100:.2f}%, Test Loss: {avg_loss:.4f}')
        return accuracy, avg_loss
