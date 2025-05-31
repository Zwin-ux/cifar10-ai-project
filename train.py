import argparse
import torch
import torch.optim as optim
import torch.nn as nn # Keep for CV
import yaml
import os

# Modular imports - CV
from models.vision import SimpleCNN
from datasets import load_cifar10
from trainers import CVTrainer

# Modular imports - NLP
from models.nlp import load_bert_classifier
from datasets import load_sst2_dataset
from trainers import NLPTrainer
from transformers import AdamW, get_linear_schedule_with_warmup # Common for BERT

from utils import plot_train_val_curves

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = None, None # Tokenizer for NLP
    trainloader, valloader, testloader = None, None, None
    criterion, optimizer, scheduler = None, None, None
    trainer_results = None

    # --- Dataset Loading ---
    print(f"Loading {args.dataset_name} dataset (type: {args.dataset_type})...")
    if args.dataset_type == 'cv':
        if args.dataset_name == 'cifar10':
            trainloader, valloader, testloader = load_cifar10(
                batch_size=args.batch_size,
                val_split=args.val_split if hasattr(args, 'val_split') else 0.1, # CV specific
                num_workers=args.num_workers
            )
        else:
            raise ValueError(f"Unsupported CV dataset: {args.dataset_name}")
    elif args.dataset_type == 'nlp':
        # NLP models often need tokenizer first
        print(f"Loading tokenizer for {args.model_name}...")
        # This assumes load_bert_classifier returns model and tokenizer, adjust if only tokenizer is needed here
        # For NLP, num_labels is crucial and should come from config or be inferred.
        num_labels = args.num_labels if hasattr(args, 'num_labels') and args.num_labels is not None else 2
        _, tokenizer = load_bert_classifier(model_name=args.model_name, num_labels=num_labels)

        if args.dataset_name == 'sst2':
            trainloader, valloader, testloader = load_sst2_dataset(
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length if hasattr(args, 'max_length') else 128
            )
        else:
            raise ValueError(f"Unsupported NLP dataset: {args.dataset_name}")
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")
    print("Dataset loaded.")

    # --- Model Initialization ---
    print(f"Initializing {args.model_name} model (type: {args.model_type})...")
    if args.model_type == 'simplecnn': # CV
        model = SimpleCNN()
    elif args.model_type == 'bert_classifier': # NLP
        num_labels = args.num_labels if hasattr(args, 'num_labels') and args.num_labels is not None else 2
        if model is None: # Check if model is already loaded (e.g. with tokenizer)
             model, _ = load_bert_classifier(model_name=args.model_name, num_labels=num_labels)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    # --- Optimizer and Scheduler ---
    if args.dataset_type == 'cv':
        criterion = nn.CrossEntropyLoss() # CV specific
        momentum = args.momentum if hasattr(args, 'momentum') and args.momentum is not None else 0.9
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=momentum)
        # Example scheduler, consider making it configurable
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    elif args.dataset_type == 'nlp':
        criterion = None # Often handled inside HuggingFace model's forward pass if labels are provided
        optimizer = AdamW(model.parameters(), lr=args.lr)
        num_training_steps = args.epochs * len(trainloader)
        # Default num_warmup_steps to 0, can be made configurable
        num_warmup_steps = args.num_warmup_steps if hasattr(args, 'num_warmup_steps') and args.num_warmup_steps is not None else 0
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        raise ValueError(f"Unsupported dataset_type for optimizer/scheduler setup: {args.dataset_type}")

    # --- Trainer Initialization ---
    print(f"Initializing {args.dataset_type.upper()}Trainer...")
    if args.dataset_type == 'cv':
        patience = args.patience if hasattr(args, 'patience') and args.patience is not None else 7
        trainer = CVTrainer(
            model=model, trainloader=trainloader, valloader=valloader, testloader=testloader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device,
            epochs=args.epochs, patience=patience,
            model_save_path=args.save_path
        )
    elif args.dataset_type == 'nlp':
        trainer = NLPTrainer(
            model=model, train_dataloader=trainloader, eval_dataloader=valloader, test_dataloader=testloader,
            optimizer=optimizer, scheduler=scheduler, device=device, epochs=args.epochs,
            model_save_path=args.save_path
        )
    else:
        raise ValueError(f"Unsupported dataset_type for trainer: {args.dataset_type}")

    # --- Training & Testing ---
    print("Starting training process...")
    trainer_results = trainer.train()
    print("Training finished.")

    print("Starting testing process...")
    test_acc, test_loss = trainer.test()
    print(f"Test Results - Accuracy: {test_acc*100:.2f}%, Loss: {test_loss:.4f}")

    # --- Plotting ---
    if args.plot_curves:
        if args.dataset_type == 'cv' and trainer_results:
            train_losses, train_accs, val_losses, val_accs = trainer_results
            print("Plotting CV curves...")
            plot_train_val_curves(train_losses, val_losses, train_accs, val_accs, title_suffix=f"{args.model_name} on {args.dataset_name}")
        elif args.dataset_type == 'nlp' and trainer_results:
            train_losses, eval_losses, eval_accuracies = trainer_results # NLPTrainer returns these three
            print("Plotting NLP curves (Train Loss, Eval Loss, Eval Accuracy)...")
            # plot_train_val_curves can be adapted or a new function for NLP plots
            # For now, let's plot train_loss vs eval_loss and a separate one for eval_accuracy
            # This is a placeholder for more specific NLP plotting.
            plt = __import__('matplotlib.pyplot') # Dynamic import for plotting
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(eval_losses, label='Eval Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'NLP Loss Curves: {args.model_name} on {args.dataset_name}')

            plt.subplot(1, 2, 2)
            plt.plot(eval_accuracies, label='Eval Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title(f'NLP Eval Accuracy: {args.model_name} on {args.dataset_name}')
            plt.tight_layout()
            plt.show()
        else:
            print("Plotting not configured or results not available for the specified type.")

    print(f"Best model was saved by the trainer at: {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modular AI Training Script')
    parser.add_argument('--config_file', type=str, default=None, help='Path to YAML configuration file')

    temp_args, _ = parser.parse_known_args()

    defaults = {
        'epochs': 1, 'batch_size': 64, 'lr': 0.001, 'use_cuda': True, 'num_workers': 2,
        'save_path': 'artifacts/default_best.pth', 'plot_curves': True,
        'model_type': 'simplecnn', 'dataset_type': 'cv',
        'model_name': 'simplecnn', 'dataset_name': 'cifar10',
        'val_split': 0.1, 'patience': 7, 'momentum': 0.9, # CV
        'max_length': 128, 'num_labels': 2, 'num_warmup_steps': 0 # NLP
    }

    if temp_args.config_file:
        try:
            with open(temp_args.config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                defaults.update(yaml_config)
                print(f"Loaded configuration from {temp_args.config_file}")
        except FileNotFoundError:
            print(f"Warning: Config file {temp_args.config_file} not found.")
        except Exception as e:
            print(f"Error loading config file {temp_args.config_file}: {e}.")

    parser.add_argument('--model_type', type=str, help='Type of model')
    parser.add_argument('--dataset_type', type=str, help='Type of dataset')
    parser.add_argument('--model_name', type=str, help='Name of the model architecture')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')

    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Input batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--use_cuda', type=lambda x: (str(x).lower() == 'true'), help='Enable CUDA training')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('--save_path', type=str, help='Path to save the best model')
    parser.add_argument('--plot_curves', type=lambda x: (str(x).lower() == 'true'), help='Plot training/validation curves')

    # CV specific
    parser.add_argument('--val_split', type=float, help='Proportion of training data for validation (CV)')
    parser.add_argument('--patience', type=int, help='Patience for early stopping (CV)')
    parser.add_argument('--momentum', type=float, help='Momentum for SGD optimizer (CV)')

    # NLP specific
    parser.add_argument('--max_length', type=int, help='Max sequence length for tokenizer (NLP)')
    parser.add_argument('--num_labels', type=int, help='Number of labels for classification (NLP)')
    parser.add_argument('--num_warmup_steps', type=int, help='Number of warmup steps for scheduler (NLP)')

    parser.set_defaults(**defaults) # Set defaults after all args are added
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory for save_path: {save_dir}")

    main(args)
