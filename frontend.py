import streamlit as st
import pandas as pd
from inference import SentimentInference

# Configure page
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭")

# Initialize session state
if 'inference' not in st.session_state:
    st.session_state.inference = None
if 'options' not in st.session_state:
    st.session_state.options = None

# Header
st.title("🎭 Sentiment Analysis")

# Initialize inference if not already done
if st.session_state.inference is None:
    with st.spinner("Loading model..."):
        try:
            st.session_state.inference = SentimentInference()
            st.session_state.options = st.session_state.inference.get_available_options()
            st.session_state.inference.download_model_from_s3()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

# Get options
options = st.session_state.options

# Input form
with st.form("prediction_form"):
    # Text input
    text_input = st.text_area("Enter text:", placeholder="Type your message here...", height=100)

    # Dropdowns
    col1, col2, col3 = st.columns(3)

    with col1:
        time_choice = st.selectbox("Time of Day:", options['time_options'])

    with col2:
        age_choice = st.selectbox("Age Group:", options['age_options'])

    with col3:
        country_choice = st.selectbox("Country:", options['country_options'])

    # Submit button
    submitted = st.form_submit_button("Analyze Sentiment")

# Show results
if submitted and text_input.strip():
    with st.spinner("Analyzing..."):
        try:
            result = st.session_state.inference.predict(text_input, time_choice, age_choice, country_choice)

            # Display result
            sentiment = result['predicted_sentiment'].upper()
            confidence = result['confidence']

            # Color based on sentiment
            if sentiment == 'POSITIVE':
                st.success(f"**{sentiment}** (Confidence: {confidence:.1%})")
            elif sentiment == 'NEGATIVE':
                st.error(f"**{sentiment}** (Confidence: {confidence:.1%})")
            else:
                st.warning(f"**{sentiment}** (Confidence: {confidence:.1%})")

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif submitted and not text_input.strip():
    st.warning("Please enter some text to analyze.")