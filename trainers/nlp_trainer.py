import torch
from tqdm.auto import tqdm # For progress bars
import os # Ensure this is present

class NLPTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, optimizer, scheduler, device, epochs, model_save_path='best_nlp_model.pth'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader # Added
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.best_eval_accuracy = 0.0

        self.train_losses = []
        self.eval_losses = [] # if you want to track eval loss
        self.eval_accuracies = []


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_dataloader, desc="Training"):
            # Ensure batch items are tensors before moving to device.
            # HuggingFace datasets usually return dicts of tensors.
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Some models might not expect token_type_ids if not provided by tokenizer
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]

            outputs = self.model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step() # Depending on scheduler type, this might be per batch or per epoch
            self.optimizer.zero_grad()

        avg_train_loss = total_loss / len(self.train_dataloader)
        self.train_losses.append(avg_train_loss)
        return avg_train_loss

    def evaluate(self, dataloader, desc="Evaluating"): # Added desc parameter
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"]

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs.loss
            total_eval_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_eval_accuracy += (predictions == batch["labels"]).sum().item()

        avg_val_accuracy = total_eval_accuracy / len(dataloader.dataset)
        avg_val_loss = total_eval_loss / len(dataloader)
        return avg_val_accuracy, avg_val_loss


    def train(self):
        print(f"Starting NLP training for {self.epochs} epochs on {self.device}...")
        for epoch in range(self.epochs):
            avg_train_loss = self.train_epoch()
            eval_accuracy, eval_loss = self.evaluate(self.eval_dataloader, desc="Evaluating Validation Set")

            self.eval_accuracies.append(eval_accuracy)
            self.eval_losses.append(eval_loss)

            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")

            if eval_accuracy > self.best_eval_accuracy:
                self.best_eval_accuracy = eval_accuracy
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"New best model saved with accuracy: {self.best_eval_accuracy:.4f} to {self.model_save_path}")

        print("Finished NLP Training.")
        print(f"Best Eval Accuracy: {self.best_eval_accuracy:.4f}")

        if self.model_save_path and os.path.exists(self.model_save_path):
             self.model.load_state_dict(torch.load(self.model_save_path))
             print(f"Loaded best model from {self.model_save_path} for final testing.")

        return self.train_losses, self.eval_losses, self.eval_accuracies


    def test(self):
        print("Testing NLP model...")
        # Use the dedicated test_dataloader
        test_accuracy, test_loss = self.evaluate(self.test_dataloader, desc="Evaluating Test Set")
        print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
        return test_accuracy, test_loss
