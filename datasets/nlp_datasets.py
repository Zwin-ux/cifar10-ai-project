from datasets import load_dataset
from torch.utils.data import DataLoader

def load_sst2_dataset(tokenizer, batch_size=32, max_length=128):
    dataset = load_dataset('glue', 'sst2')

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    # SST-2 doesn't have a readily available test set in 'datasets', often evaluated on validation or a specific test server.
    # For simplicity, we'll use the validation set as the test set here.
    test_dataset = tokenized_datasets["validation"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # Using val as test

    return train_dataloader, eval_dataloader, test_dataloader
