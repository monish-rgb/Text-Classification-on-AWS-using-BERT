import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertModel, DistilBertTokenizer,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import logging
import sys
import os

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def train(args):
    """
    Main training function to fine-tune the DistilBERT model.
    """
    # 1. Load the preprocessed dataset from the input channel
    # SageMaker maps the 'train' channel from S3 to this local directory.
    train_input_data_path = os.path.join(args.train, 'train.pth')
    test_input_data_path = os.path.join(args.test, 'test.pth')
    logger.info(f"Loading preprocessed data from: {train_input_data_path}")
    logger.info(f"Loading preprocessed data from: {test_input_data_path}")
    try:
        train_dataset = torch.load(train_input_data_path)
        val_dataset = torch.load(test_input_data_path)
    except FileNotFoundError:
        logger.error(f"Error: The training or test data file was not found at.")
        sys.exit(1)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")


    # 3. Load the pre-trained DistilBERT model
    # The number of labels needs to be specified.
    logger.info(f"Loading model 'distilbert-base-uncased' for text classification with {4} labels.")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=4
    )

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # 4. Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataset) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 5. Start the training loop
    logger.info("***** Starting training *****")
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        for batch in train_dataset:
            # Move batch to the correct device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            targets = batch[2].to(device)

            # Clear previous gradients
            model.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataset)
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        logger.info(f"  Average training loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        model.eval()  # Set model to evaluation mode
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in val_dataset:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad(): # No need to calculate gradients for validation
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()

            # Move logits and labels to CPU for metric calculation
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            # Calculate accuracy
            preds = logits.argmax(axis=1)
            total_eval_accuracy += accuracy_score(label_ids, preds)

        avg_val_accuracy = total_eval_accuracy / len(val_dataset)
        avg_val_loss = total_eval_loss / len(val_dataset)
        logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"  Validation Accuracy: {avg_val_accuracy:.4f}")

    # 6. Save the final model
    # The model is saved to the directory specified by SageMaker's SM_MODEL_DIR environment variable.
    logger.info("Training complete. Saving the model.")
    model.save_pretrained(args.model_dir)

    # Instantiate and save the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.save_pretrained(args.model_dir)

if __name__ == "__main__":
    # Parser for command-line arguments passed by the SageMaker Estimator
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--train_batch_size",type=int,default=4)
    parser.add_argument("--valid_batch_size",type=int,default=2)
    parser.add_argument("--learning_rate",type=float,default=5e-5)

    # --- SageMaker Environment Paths ---
    # These are automatically provided by the SageMaker environment.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    
    args, _ = parser.parse_known_args()

    train(args)








