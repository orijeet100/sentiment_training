import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Custom Dataset for BERT sentiment analysis with categorical features."""

    def __init__(self, texts, labels, time_encoded, age_encoded, country_encoded, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.time_encoded = time_encoded
        self.age_encoded = age_encoded
        self.country_encoded = country_encoded
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'time_feature': torch.tensor(self.time_encoded[idx], dtype=torch.long),
            'age_feature': torch.tensor(self.age_encoded[idx], dtype=torch.long),
            'country_feature': torch.tensor(self.country_encoded[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTSentimentModel(torch.nn.Module):
    """BERT model with additional categorical features."""

    def __init__(self, n_classes, n_time_classes, n_age_classes, n_country_classes):
        super(BERTSentimentModel, self).__init__()

        # BERT backbone
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=n_classes,
            output_hidden_states=True
        )

        # Embedding layers for categorical features
        self.time_embedding = torch.nn.Embedding(n_time_classes, 32)
        self.age_embedding = torch.nn.Embedding(n_age_classes, 32)
        self.country_embedding = torch.nn.Embedding(n_country_classes, 64)

        # Combined classifier
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768 + 32 + 32 + 64, n_classes)

    def forward(self, input_ids, attention_mask, time_feature, age_feature, country_feature):
        # Get BERT outputs
        bert_outputs = self.bert.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.last_hidden_state[:, 0]  # [CLS] token

        # Get categorical embeddings
        time_emb = self.time_embedding(time_feature)
        age_emb = self.age_embedding(age_feature)
        country_emb = self.country_embedding(country_feature)

        # Combine features
        combined_features = torch.cat([pooled_output, time_emb, age_emb, country_emb], dim=1)
        combined_features = self.dropout(combined_features)

        # Final classification
        logits = self.classifier(combined_features)
        return logits


class SentimentModelTrainer:
    """Trainer class for BERT sentiment analysis."""

    def __init__(self, mlflow_tracking_uri="http://ec2-3-141-41-8.us-east-2.compute.amazonaws.com:5000/"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Set up MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("sentiment_analysis_bert")

        logger.info(f"Using device: {self.device}")
        logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

    def load_and_prepare_data(self, data_path="processed_data/train_processed.csv"):
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")

        # Load data
        df = pd.read_csv(data_path)

        # Select required features
        required_columns = ['text', 'Time of Tweet', 'Age of User', 'Country', 'sentiment']
        df = df[required_columns].dropna()

        # Encode categorical features
        self.label_encoder = LabelEncoder()
        self.time_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()
        self.country_encoder = LabelEncoder()

        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        df['time_encoded'] = self.time_encoder.fit_transform(df['Time of Tweet'])
        df['age_encoded'] = self.age_encoder.fit_transform(df['Age of User'])
        df['country_encoded'] = self.country_encoder.fit_transform(df['Country'])

        # Store class information
        self.n_classes = len(self.label_encoder.classes_)
        self.n_time_classes = len(self.time_encoder.classes_)
        self.n_age_classes = len(self.age_encoder.classes_)
        self.n_country_classes = len(self.country_encoder.classes_)

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        logger.info(f"Time categories: {len(self.time_encoder.classes_)}")
        logger.info(f"Age categories: {len(self.age_encoder.classes_)}")
        logger.info(f"Country categories: {len(self.country_encoder.classes_)}")

        return df

    def create_data_loaders(self, df, test_size=0.2, batch_size=16):
        """Create train and test data loaders."""
        logger.info("Creating data loaders...")

        # Train-test split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment_encoded']
        )

        # Create datasets
        train_dataset = SentimentDataset(
            train_df['text'],
            train_df['sentiment_encoded'].values,
            train_df['time_encoded'].values,
            train_df['age_encoded'].values,
            train_df['country_encoded'].values,
            self.tokenizer
        )

        test_dataset = SentimentDataset(
            test_df['text'],
            test_df['sentiment_encoded'].values,
            test_df['time_encoded'].values,
            test_df['age_encoded'].values,
            test_df['country_encoded'].values,
            self.tokenizer
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        return train_loader, test_loader, train_df, test_df

    def create_model(self):
        """Create the BERT model."""
        model = BERTSentimentModel(
            n_classes=self.n_classes,
            n_time_classes=self.n_time_classes,
            n_age_classes=self.n_age_classes,
            n_country_classes=self.n_country_classes
        ).to(self.device)

        return model

    def train_model(self, model, train_loader, test_loader, epochs=3, learning_rate=2e-5):
        """Train the model."""
        logger.info("Starting training...")

        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", train_loader.batch_size)
            mlflow.log_param("model_type", "DistilBERT_with_categorical")

            for epoch in range(epochs):
                # Training phase
                model.train()
                total_train_loss = 0
                train_correct = 0
                train_total = 0

                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} - Training')
                for batch in train_pbar:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    time_feature = batch['time_feature'].to(self.device)
                    age_feature = batch['age_feature'].to(self.device)
                    country_feature = batch['country_feature'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    logits = model(input_ids, attention_mask, time_feature, age_feature, country_feature)
                    loss = criterion(logits, labels)

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Track metrics
                    total_train_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * train_correct / train_total:.2f}%'
                    })

                # Calculate training metrics
                avg_train_loss = total_train_loss / len(train_loader)
                train_accuracy = train_correct / train_total

                # Evaluation phase
                test_accuracy, test_loss, test_metrics = self.evaluate_model(model, test_loader, criterion)

                # Log metrics to MLflow
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("test_loss", test_loss, step=epoch)
                mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)
                mlflow.log_metric("test_precision", test_metrics['precision'], step=epoch)
                mlflow.log_metric("test_recall", test_metrics['recall'], step=epoch)
                mlflow.log_metric("test_f1", test_metrics['f1'], step=epoch)

                logger.info(f"Epoch {epoch + 1}/{epochs}")
                logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
                logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
                logger.info(f"Test F1: {test_metrics['f1']:.4f}")

            # Save model
            model_path = "models/bert_sentiment_model.pth"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': self.label_encoder,
                'time_encoder': self.time_encoder,
                'age_encoder': self.age_encoder,
                'country_encoder': self.country_encoder,
                'model_config': {
                    'n_classes': self.n_classes,
                    'n_time_classes': self.n_time_classes,
                    'n_age_classes': self.n_age_classes,
                    'n_country_classes': self.n_country_classes
                }
            }, model_path)

            # Log model to MLflow
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact(model_path)

            logger.info(f"Model saved to {model_path}")
            logger.info("Training completed!")

        return model

    def evaluate_model(self, model, test_loader, criterion):
        """Evaluate the model."""
        model.eval()
        total_test_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                time_feature = batch['time_feature'].to(self.device)
                age_feature = batch['age_feature'].to(self.device)
                country_feature = batch['country_feature'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = model(input_ids, attention_mask, time_feature, age_feature, country_feature)
                loss = criterion(logits, labels)

                total_test_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        avg_test_loss = total_test_loss / len(test_loader)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return test_accuracy, avg_test_loss, metrics

    def run_training_pipeline(self, data_path="processed_data/train_processed.csv", epochs=3, batch_size=16):
        """Run the complete training pipeline."""
        logger.info("Starting BERT sentiment analysis training pipeline...")

        # Load and prepare data
        df = self.load_and_prepare_data(data_path)

        # Create data loaders
        train_loader, test_loader, train_df, test_df = self.create_data_loaders(df, batch_size=batch_size)

        # Create model
        model = self.create_model()

        # Train model
        trained_model = self.train_model(model, train_loader, test_loader, epochs=epochs)

        logger.info("Training pipeline completed successfully!")
        return trained_model


# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = SentimentModelTrainer()

    # Run training pipeline
    try:
        model = trainer.run_training_pipeline(
            data_path="processed_data/train_processed.csv",
            epochs=3,
            batch_size=16
        )
        logger.info("BERT sentiment model training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise