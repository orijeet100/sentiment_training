# Sentiment Analysis with BERT and AWS SageMaker

A comprehensive sentiment analysis system that uses BERT (DistilBERT) for text classification with additional categorical features (time, age, country) and deploys to AWS SageMaker for production inference.

## 🚀 Features

- **Advanced BERT Model**: Uses DistilBERT with custom categorical feature embeddings
- **Multi-feature Analysis**: Incorporates text, time of tweet, user age, and country
- **AWS SageMaker Deployment**: Production-ready model deployment
- **Streamlit Web Interface**: User-friendly web application for sentiment analysis
- **MLflow Integration**: Model versioning and experiment tracking
- **Comprehensive EDA**: Exploratory data analysis with visualizations
- **Robust Preprocessing**: Handles encoding issues, data cleaning, and validation

## 📁 Project Structure

```
├── data/                          # Raw dataset files
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── processed_data/                # Preprocessed data
├── model_sagemaker/              # SageMaker deployment package
│   ├── code/                     # Inference code and encoders
│   ├── model_sagemaker.pth       # Trained model weights
│   └── model_sagemaker.tar.gz    # SageMaker deployment package
├── eda_plots/                    # Exploratory data analysis visualizations
├── model_building.py             # BERT model training pipeline
├── preprocessing.py              # Data preprocessing and cleaning
├── streamlit_interface_sagemaker.py  # Web interface
├── deploy_model_sagemaker.py     # SageMaker deployment script
├── local_testing.py              # Local model testing
├── sagemaker_inference.py        # SageMaker inference handler
├── eda.py                        # Exploratory data analysis
├── requirements.txt              # Python dependencies
└── bert_sentiment_model.pth      # Local model file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sentiment_Analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **AWS Configuration** (for SageMaker deployment)
   ```bash
   aws configure
   ```

## 📊 Data Preprocessing

The preprocessing pipeline handles:
- Encoding detection and handling
- Text cleaning and normalization
- Categorical feature encoding
- Data validation and filtering
- Train/test split preparation

```bash
python preprocessing.py
```

## 🧠 Model Training

The BERT model incorporates:
- **DistilBERT backbone** for text processing
- **Categorical embeddings** for time, age, and country features
- **Combined classifier** with dropout for regularization
- **MLflow tracking** for experiment management

```bash
python model_building.py
```

## 🚀 Deployment

### Local Testing
```bash
python local_testing.py
```

### AWS SageMaker Deployment
```bash
python deploy_model_sagemaker.py
```

## 🌐 Web Interface

Launch the Streamlit web application:
```bash
streamlit run streamlit_interface_sagemaker.py
```

Features:
- Real-time sentiment analysis
- Interactive input forms
- Probability visualization
- Performance metrics
- Batch processing capabilities

## 📈 Model Architecture

### BERT + Categorical Features
- **Text Processing**: DistilBERT with 768-dimensional embeddings
- **Categorical Features**: 
  - Time of Tweet: 32-dimensional embedding
  - Age of User: 32-dimensional embedding  
  - Country: 64-dimensional embedding
- **Combined Classifier**: 896-dimensional input → sentiment classes
- **Regularization**: Dropout (0.3) for overfitting prevention

### Training Configuration
- **Optimizer**: AdamW with learning rate 2e-5
- **Scheduler**: Linear warmup with cosine decay
- **Batch Size**: 16 (configurable)
- **Epochs**: 3 (configurable)
- **Loss**: Cross-entropy

## 🔧 Configuration

### Environment Variables
- `AWS_DEFAULT_REGION`: AWS region for SageMaker
- `MLFLOW_TRACKING_URI`: MLflow server URI
- `SAGEMAKER_ENDPOINT_NAME`: SageMaker endpoint name

### Model Parameters
- **Max Sequence Length**: 128 tokens
- **Embedding Dimensions**: Time (32), Age (32), Country (64)
- **Dropout Rate**: 0.3
- **Learning Rate**: 2e-5

## 📊 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed error analysis
- **Inference Time**: Real-time performance measurement

## 🛡️ Error Handling

- **Encoding Detection**: Automatic CSV encoding detection
- **Data Validation**: Comprehensive input validation
- **AWS Error Handling**: Graceful SageMaker endpoint errors
- **Fallback Options**: Default values for missing data

## 🔍 Exploratory Data Analysis

Run comprehensive EDA:
```bash
python eda.py
```

Generates visualizations for:
- Sentiment distribution
- Text length analysis
- Categorical feature distributions
- Word clouds
- Correlation analysis

## 📝 API Usage

### SageMaker Endpoint
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')
payload = {
    'text': 'I love this product!',
    'time_of_tweet': 'morning',
    'age_of_user': '21-30',
    'country': 'USA'
}

response = runtime.invoke_endpoint(
    EndpointName='sentiment-endpoint-test',
    ContentType='application/json',
    Body=json.dumps(payload)
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers** for BERT implementation
- **AWS SageMaker** for model deployment
- **Streamlit** for web interface
- **MLflow** for experiment tracking

## 📞 Support

For questions or issues:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error logs and system information

---

**Note**: Ensure you have sufficient AWS permissions for SageMaker deployment and S3 access for model artifacts. 
