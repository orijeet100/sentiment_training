import streamlit as st
import boto3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .positive-result {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #155724;
    }

    .negative-result {
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #721c24;
    }

    .neutral-result {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        color: #856404;
    }

    .info-box {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class SageMakerSentimentAnalyzer:
    def __init__(self, endpoint_name='sentiment-endpoint', region='us-east-2'):
        self.endpoint_name = endpoint_name
        self.region = region
        self.runtime_client = None
        self.s3_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize AWS clients."""
        try:
            self.runtime_client = boto3.client('sagemaker-runtime', region_name=self.region)
            self.s3_client = boto3.client('s3', region_name=self.region)
            logger.info("AWS clients initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AWS clients: {str(e)}")
            st.error("⚠️ AWS credentials not configured. Please run 'aws configure'")

    def test_endpoint_connection(self):
        """Test if the SageMaker endpoint is accessible."""
        try:
            # Try a simple test call
            test_payload = {
                'text': 'test',
                'time_of_tweet': 'morning',
                'age_of_user': '21-30',
                'country': 'USA'
            }

            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(test_payload)
            )

            return True, "Endpoint is accessible"

        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg:
                return False, f"Endpoint '{self.endpoint_name}' not found"
            elif "403" in error_msg:
                return False, "Access denied. Check your AWS credentials and permissions"
            else:
                return False, f"Connection error: {error_msg}"

    def get_dropdown_options(self, bucket_name='mlflow-bucket-ori'):
        """Load dropdown options from S3 processed data."""
        try:
            # Try to download processed data from S3
            temp_file = 'temp_processed_data.csv'
            self.s3_client.download_file(
                bucket_name,
                'processed_data/train_processed.csv',
                temp_file
            )

            df = pd.read_csv(temp_file)

            options = {
                'time_options': sorted(df['Time of Tweet'].dropna().unique().tolist()),
                'age_options': sorted(df['Age of User'].dropna().unique().tolist()),
                'country_options': sorted(df['Country'].dropna().unique().tolist()[:50])  # Limit for performance
            }

            # Clean up temp file
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)

            return options

        except Exception as e:
            logger.warning(f"Could not load options from S3: {str(e)}")
            # Return fallback options
            return {
                'time_options': ['morning', 'noon', 'night'],
                'age_options': ['0-20', '21-30', '31-45', '46-60'],
                'country_options': ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'India']
            }

    def predict_sentiment(self, text, time_of_tweet, age_of_user, country):
        """Make sentiment prediction using SageMaker endpoint."""
        try:
            payload = {
                'text': text,
                'time_of_tweet': time_of_tweet,
                'age_of_user': age_of_user,
                'country': country
            }

            # Measure inference time
            start_time = time.time()

            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            inference_time = time.time() - start_time

            # Parse response
            result = json.loads(response['Body'].read().decode())
            result['inference_time'] = inference_time

            return result, None

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Prediction error: {error_msg}")
            return None, error_msg


def initialize_app():
    """Initialize the app and load necessary data."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SageMakerSentimentAnalyzer()

    if 'options' not in st.session_state:
        with st.spinner("Loading dropdown options..."):
            st.session_state.options = st.session_state.analyzer.get_dropdown_options()

    if 'endpoint_status' not in st.session_state:
        with st.spinner("Testing endpoint connection..."):
            is_connected, status_msg = st.session_state.analyzer.test_endpoint_connection()
            st.session_state.endpoint_status = (is_connected, status_msg)


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎭 AI Sentiment Analysis</h1>
        <p>Powered by AWS SageMaker • Advanced NLP with Contextual Features</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with app information."""
    with st.sidebar:
        st.header("📊 App Information")

        # Endpoint status
        is_connected, status_msg = st.session_state.endpoint_status
        if is_connected:
            st.success("✅ SageMaker Endpoint: Connected")
        else:
            st.error(f"❌ SageMaker Endpoint: {status_msg}")

        # Options info
        options = st.session_state.options
        st.markdown("""
        <div class="info-box">
            <h4>📋 Available Options</h4>
            <ul>
                <li><strong>Time Periods:</strong> """ + str(len(options['time_options'])) + """</li>
                <li><strong>Age Groups:</strong> """ + str(len(options['age_options'])) + """</li>
                <li><strong>Countries:</strong> """ + str(len(options['country_options'])) + """</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Architecture info
        st.markdown("""
        <div class="info-box">
            <h4>🏗️ Architecture</h4>
            <ul>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>Model:</strong> DistilBERT + Custom Features</li>
                <li><strong>Deployment:</strong> AWS SageMaker</li>
                <li><strong>Instance:</strong> ml.m5.large</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Sample texts
        st.subheader("💡 Sample Texts")
        samples = [
            "I absolutely love this product! Best purchase ever!",
            "This is terrible. Completely disappointed.",
            "It's okay, nothing special but decent quality.",
            "Feeling great today! Life is wonderful!",
            "I'm frustrated with this situation."
        ]

        for i, sample in enumerate(samples):
            if st.button(f"📝 Sample {i + 1}", key=f"sample_{i}", help=sample[:50] + "..."):
                st.session_state.sample_text = sample


def render_input_form():
    """Render the main input form."""
    st.subheader("📝 Enter Text and Context Information")

    options = st.session_state.options

    # Main input form
    with st.form("sentiment_analysis_form", clear_on_submit=False):
        # Text input
        default_text = st.session_state.get('sample_text', '')
        text_input = st.text_area(
            "Text to analyze:",
            value=default_text,
            placeholder="Enter your message here... (e.g., 'I love this new feature!')",
            height=120,
            max_chars=1000,
            help="Enter any text you want to analyze for sentiment"
        )

        # Context inputs in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            time_choice = st.selectbox(
                "⏰ Time of Day:",
                options['time_options'],
                help="When was this message posted?"
            )

        with col2:
            age_choice = st.selectbox(
                "👥 Age Group:",
                options['age_options'],
                help="Age group of the user"
            )

        with col3:
            country_choice = st.selectbox(
                "🌍 Country:",
                options['country_options'],
                help="User's country or region"
            )

        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            show_probabilities = st.checkbox("Show detailed probabilities", value=True)
            show_metrics = st.checkbox("Show performance metrics", value=True)

        # Submit button
        submitted = st.form_submit_button(
            "🔮 Analyze Sentiment",
            type="primary",
            use_container_width=True
        )

    return text_input, time_choice, age_choice, country_choice, submitted, show_probabilities, show_metrics


def render_results(result, text_input, time_choice, age_choice, country_choice, show_probabilities, show_metrics):
    """Render the analysis results."""
    sentiment = result['predicted_sentiment'].upper()
    confidence = result['confidence']
    all_probabilities = result['all_probabilities']
    inference_time = result.get('inference_time', 0)

    # Main result display
    st.subheader("🎯 Analysis Results")

    # Choose styling based on sentiment
    if sentiment == 'POSITIVE':
        result_class = "positive-result"
        emoji = "😊"
        color = "#28a745"
    elif sentiment == 'NEGATIVE':
        result_class = "negative-result"
        emoji = "😞"
        color = "#dc3545"
    else:
        result_class = "neutral-result"
        emoji = "😐"
        color = "#ffc107"

    # Main result card
    st.markdown(f"""
    <div class="{result_class}">
        <h2>{emoji} {sentiment}</h2>
        <h3>Confidence: {confidence:.1%}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Detailed analysis
    col1, col2 = st.columns([3, 2])

    with col1:
        if show_probabilities:
            st.subheader("📊 Probability Distribution")

            # Create probability chart
            prob_df = pd.DataFrame([
                {"Sentiment": k.title(), "Probability": v, "Color":
                    "#28a745" if k.lower() == "positive" else
                    "#dc3545" if k.lower() == "negative" else "#ffc107"}
                for k, v in all_probabilities.items()
            ])

            fig = px.bar(
                prob_df,
                x="Sentiment",
                y="Probability",
                color="Color",
                color_discrete_map={c: c for c in prob_df["Color"]},
                title="Sentiment Probability Distribution"
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Sentiment",
                yaxis_title="Probability",
                yaxis=dict(tickformat=".1%")
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 Input Summary")

        # Input details
        st.markdown(f"""
        <div class="metric-card">
            <strong>📝 Text:</strong><br>
            <em>"{text_input[:100]}{'...' if len(text_input) > 100 else ''}"</em>
        </div>

        <div class="metric-card">
            <strong>⏰ Time:</strong> {time_choice}<br>
            <strong>👥 Age:</strong> {age_choice}<br>
            <strong>🌍 Country:</strong> {country_choice}
        </div>

        <div class="metric-card">
            <strong>📏 Text Length:</strong> {len(text_input)} characters<br>
            <strong>📊 Word Count:</strong> {len(text_input.split())} words
        </div>
        """, unsafe_allow_html=True)

        if show_metrics:
            st.markdown(f"""
            <div class="metric-card">
                <strong>⚡ Inference Time:</strong> {inference_time:.2f}s<br>
                <strong>🎯 Model:</strong> DistilBERT + Features<br>
                <strong>☁️ Powered by:</strong> AWS SageMaker
            </div>
            """, unsafe_allow_html=True)

    # Confidence interpretation
    st.subheader("💡 Interpretation")

    if confidence > 0.8:
        confidence_msg = "Very confident prediction. The model is highly certain about this sentiment."
        confidence_color = "success"
    elif confidence > 0.6:
        confidence_msg = "Moderately confident prediction. The sentiment is fairly clear."
        confidence_color = "warning"
    else:
        confidence_msg = "Low confidence prediction. The sentiment may be ambiguous or mixed."
        confidence_color = "error"

    if confidence_color == "success":
        st.success(confidence_msg)
    elif confidence_color == "warning":
        st.warning(confidence_msg)
    else:
        st.error(confidence_msg)


def main():
    """Main application function."""
    # Initialize app
    initialize_app()

    # Check if endpoint is connected
    is_connected, status_msg = st.session_state.endpoint_status
    if not is_connected:
        st.error(f"❌ Cannot connect to SageMaker endpoint: {status_msg}")
        st.info("Please make sure:")
        st.markdown("""
        - Your AWS credentials are configured (`aws configure`)
        - The SageMaker endpoint 'sentiment-endpoint' is deployed and running
        - You have the necessary permissions to invoke the endpoint
        """)
        st.stop()

    # Render UI
    render_header()
    render_sidebar()

    # Main content
    text_input, time_choice, age_choice, country_choice, submitted, show_probabilities, show_metrics = render_input_form()

    # Process submission
    if submitted:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return

        # Show analysis progress
        with st.spinner("🧠 Analyzing sentiment with AI..."):
            result, error = st.session_state.analyzer.predict_sentiment(
                text_input.strip(),
                time_choice,
                age_choice,
                country_choice
            )

        if error:
            st.error(f"❌ Analysis failed: {error}")
            st.info("Please check the CloudWatch logs for more details.")
        else:
            render_results(result, text_input, time_choice, age_choice, country_choice, show_probabilities,
                           show_metrics)

            # Add to session history (optional)
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []

            st.session_state.analysis_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
                'sentiment': result['predicted_sentiment'],
                'confidence': result['confidence'],
                'time': time_choice,
                'age': age_choice,
                'country': country_choice
            })

    # Show recent history
    if st.session_state.get('analysis_history'):
        with st.expander("📚 Recent Analysis History"):
            history_df = pd.DataFrame(st.session_state.analysis_history[-10:])  # Last 10 analyses
            st.dataframe(history_df, use_container_width=True)


if __name__ == "__main__":
    main()