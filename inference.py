import torch
import boto3
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import logging
from transformers import DistilBertTokenizer
from pathlib import Path
import os
import json
import pickle
from typing import Dict, List, Any
import tempfile
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BERTSentimentModel(torch.nn.Module):
    """BERT model with additional categorical features - same as training."""

    def __init__(self, n_classes, n_time_classes, n_age_classes, n_country_classes):
        super(BERTSentimentModel, self).__init__()

        from transformers import DistilBertForSequenceClassification

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


class SentimentInference:
    """Inference class for BERT sentiment analysis model."""

    def __init__(self,
                 s3_bucket="mlflow-bucket-ori",
                 model_s3_path="s3://mlflow-bucket-ori/897117735142510085/bde8c64873354c7aa08a1d58173ab873/artifacts/bert_sentiment_model.pth"):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.time_encoder = None
        self.age_encoder = None
        self.country_encoder = None
        self.model_config = None

        # S3 configuration
        self.s3_bucket = s3_bucket
        self.model_s3_path = model_s3_path
        self.s3_client = boto3.client('s3')

        logger.info(f"Inference initialized with device: {self.device}")
        logger.info(f"S3 bucket: {self.s3_bucket}")

    def load_dropdown_options_from_s3(self):
        """Load dropdown options from processed data in S3."""
        try:
            logger.info("Loading dropdown options from S3...")

            # Create temp file manually to avoid Windows file handle issues
            temp_file_path = os.path.join(tempfile.gettempdir(), f"train_data_{os.getpid()}.csv")

            try:
                # Download processed training data to get unique values
                self.s3_client.download_file(
                    self.s3_bucket,
                    'processed_data/train_processed.csv',
                    temp_file_path
                )

                # Load data and extract unique values
                df = pd.read_csv(temp_file_path)

                # Get unique values for dropdowns
                options = {
                    'time_options': sorted(df['Time of Tweet'].dropna().unique().tolist()),
                    'age_options': sorted(df['Age of User'].dropna().unique().tolist()),
                    'country_options': sorted(df['Country'].dropna().unique().tolist()),
                    'sentiment_labels': sorted(df['sentiment'].dropna().unique().tolist())
                }

                logger.info(f"Loaded options from S3:")
                logger.info(f"  Time options: {len(options['time_options'])} ({options['time_options'][:5]}...)")
                logger.info(f"  Age options: {len(options['age_options'])} ({options['age_options']})")
                logger.info(
                    f"  Country options: {len(options['country_options'])} ({options['country_options'][:10]}...)")
                logger.info(f"  Sentiment labels: {options['sentiment_labels']}")

                return options

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass  # Ignore cleanup errors

        except Exception as e:
            logger.error(f"Error loading options from S3: {str(e)}")
            # Return fallback options
            return {
                'time_options': ['morning', 'noon', 'night'],
                'age_options': ['0-20', '21-30', '31-45', '46-60'],
                'country_options': ['USA', 'UK', 'Canada'],
                'sentiment_labels': ['positive', 'negative', 'neutral']
            }

    def download_model_from_s3(self):
        """Download model directly from S3 path."""
        try:
            logger.info(f"Loading model from S3: {self.model_s3_path}")

            # Parse S3 path
            s3_path_parts = self.model_s3_path.replace('s3://', '').split('/')
            bucket = s3_path_parts[0]
            key = '/'.join(s3_path_parts[1:])

            # Create temp file manually to avoid Windows file handle issues
            temp_file_path = os.path.join(tempfile.gettempdir(), f"model_{os.getpid()}.pth")

            try:
                # Download model file
                self.s3_client.download_file(bucket, key, temp_file_path)

                # Load checkpoint with safe globals for sklearn objects
                torch.serialization.add_safe_globals([LabelEncoder])
                checkpoint = torch.load(temp_file_path, map_location=self.device, weights_only=False)

                # Extract components
                self.label_encoder = checkpoint['label_encoder']
                self.time_encoder = checkpoint['time_encoder']
                self.age_encoder = checkpoint['age_encoder']
                self.country_encoder = checkpoint['country_encoder']
                self.model_config = checkpoint['model_config']

                # Initialize tokenizer
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

                # Initialize and load model
                self.model = BERTSentimentModel(
                    n_classes=self.model_config['n_classes'],
                    n_time_classes=self.model_config['n_time_classes'],
                    n_age_classes=self.model_config['n_age_classes'],
                    n_country_classes=self.model_config['n_country_classes']
                ).to(self.device)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                logger.info("Model loaded successfully from S3")

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass  # Ignore cleanup errors

        except Exception as e:
            logger.error(f"Error loading model from S3: {str(e)}")
            raise

    def get_available_options(self) -> Dict[str, List[str]]:
        """Get available options for categorical features from S3 or encoders."""
        try:
            # If model is loaded, get from encoders
            if all([self.time_encoder, self.age_encoder, self.country_encoder, self.label_encoder]):
                return {
                    'time_options': self.time_encoder.classes_.tolist(),
                    'age_options': self.age_encoder.classes_.tolist(),
                    'country_options': self.country_encoder.classes_.tolist(),
                    'sentiment_labels': self.label_encoder.classes_.tolist()
                }
            else:
                # Otherwise load from S3 processed data
                return self.load_dropdown_options_from_s3()

        except Exception as e:
            logger.error(f"Error getting options: {str(e)}")
            # Fallback options
            return {
                'time_options': ['morning', 'noon', 'night'],
                'age_options': ['0-20', '21-30', '31-45', '46-60'],
                'country_options': ['USA', 'UK', 'Canada'],
                'sentiment_labels': ['positive', 'negative', 'neutral']
            }

    def preprocess_input(self, text: str, time_of_tweet: str, age_of_user: str, country: str) -> Dict[
        str, torch.Tensor]:
        """Preprocess input for model inference."""

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        # Encode categorical features
        try:
            time_encoded = self.time_encoder.transform([time_of_tweet])[0]
            age_encoded = self.age_encoder.transform([age_of_user])[0]
            country_encoded = self.country_encoder.transform([country])[0]
        except ValueError as e:
            available_options = self.get_available_options()
            raise ValueError(f"Invalid categorical value: {str(e)}. Available options: {available_options}")

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device),
            'time_feature': torch.tensor([time_encoded], dtype=torch.long).to(self.device),
            'age_feature': torch.tensor([age_encoded], dtype=torch.long).to(self.device),
            'country_feature': torch.tensor([country_encoded], dtype=torch.long).to(self.device)
        }

    def predict(self, text: str, time_of_tweet: str, age_of_user: str, country: str) -> Dict[str, Any]:
        """Make prediction for given input."""

        if self.model is None:
            raise ValueError("Model not loaded. Call download_model_from_mlflow() first.")

        try:
            # Preprocess input
            inputs = self.preprocess_input(text, time_of_tweet, age_of_user, country)

            # Make prediction
            with torch.no_grad():
                logits = self.model(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    inputs['time_feature'],
                    inputs['age_feature'],
                    inputs['country_feature']
                )

                # Get probabilities and prediction
                probabilities = torch.softmax(logits, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                predicted_class = self.label_encoder.classes_[predicted_class_idx]
                confidence = probabilities[0][predicted_class_idx].item()

                # Get all class probabilities
                all_probabilities = {}
                for i, label in enumerate(self.label_encoder.classes_):
                    all_probabilities[label] = probabilities[0][i].item()

            return {
                'predicted_sentiment': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'input_text': text,
                'input_features': {
                    'time_of_tweet': time_of_tweet,
                    'age_of_user': age_of_user,
                    'country': country
                }
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def batch_predict(self, inputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Make predictions for batch of inputs."""
        results = []
        for input_data in inputs:
            try:
                result = self.predict(
                    input_data['text'],
                    input_data['time_of_tweet'],
                    input_data['age_of_user'],
                    input_data['country']
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'input_data': input_data
                })
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize inference
    inference = SentimentInference()

    try:
        # Load dropdown options from S3
        logger.info("Loading dropdown options from S3...")
        options = inference.get_available_options()
        logger.info(f"Available options: {options}")

        # Load model from S3
        logger.info("Loading model from S3...")
        inference.download_model_from_s3()

        # Test prediction
        test_prediction = inference.predict(
            text="I love this product! It's amazing!",
            time_of_tweet=options['time_options'][0],  # Use first available option
            age_of_user=options['age_options'][0],
            country=options['country_options'][0]
        )

        logger.info(f"Test prediction: {test_prediction}")
        print(test_prediction)

    except Exception as e:
        logger.error(f"Inference test failed: {str(e)}")
        raise