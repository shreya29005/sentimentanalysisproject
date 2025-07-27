import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from textblob import TextBlob

# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Initialize stemmer
ps = nltk.stem.PorterStemmer()

# Load resources with caching
@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

@st.cache_resource
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.warning("Model files not found. Using TextBlob for sentiment analysis.")
        return None, None

def predict_sentiment_custom(text, model, vectorizer, stop_words):
    """Custom model prediction"""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    processed_text = ' '.join(text)
    
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

def predict_sentiment_textblob(text):
    """TextBlob sentiment analysis as fallback"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

def create_sentiment_card(text, sentiment, score=None, post_number=None):
    """Create colored card for sentiment display"""
    if sentiment == "Positive":
        color = "#90EE90"
    elif sentiment == "Negative":
        color = "#FFB6C1"
    else:
        color = "#FFFFE0"  # Light yellow for neutral
    
    card_html = f"""
    <div style="padding:15px; border-radius:10px; margin:10px 0; background:{color}; border-left:5px solid #333;">
        {f'<h4>Post {post_number}</h4>' if post_number else ''}
        <p style="margin:0; font-size:16px;"><strong>Sentiment:</strong> {sentiment}</p>
        {f'<p style="margin:0; font-size:14px;"><strong>Confidence Score:</strong> {score:.2f}</p>' if score else ''}
        <hr style="margin:10px 0;">
        <p style="margin:0; font-style:italic;">{text}</p>
    </div>
    """
    return card_html

def analyze_multiple_posts(captions, model, vectorizer, stop_words):
    """Analyze multiple Instagram captions"""
    results = []
    
    for i, caption in enumerate(captions):
        if caption.strip():
            if model and vectorizer:
                # Use custom model
                sentiment = predict_sentiment_custom(caption, model, vectorizer, stop_words)
                results.append({
                    'post_number': i + 1,
                    'text': caption,
                    'sentiment': sentiment,
                    'score': None
                })
            else:
                # Use TextBlob
                sentiment, score = predict_sentiment_textblob(caption)
                results.append({
                    'post_number': i + 1,
                    'text': caption,
                    'sentiment': sentiment,
                    'score': score
                })
    
    return results

def main():
    st.title("📱 Instagram Sentiment Analysis (Manual Input)")
    st.markdown("---")
    
    # Instructions
    with st.expander("📋 How to Use", expanded=True):
        st.markdown("""
        **Step 1:** Go to the Instagram profile you want to analyze
        
        **Step 2:** Copy the captions from posts you want to analyze
        
        **Step 3:** Paste them below and get instant sentiment analysis
        
        **Why manual input?** Instagram blocks automated scraping, but this method is 100% reliable!
        """)
    
    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    
    # Analysis options
    st.markdown("## Choose Analysis Method")
    option = st.selectbox(
        "Select option:",
        ["Single Post Analysis", "Multiple Posts Analysis", "Bulk Text Analysis"]
    )
    
    if option == "Single Post Analysis":
        st.markdown("### 📝 Single Post Analysis")
        caption = st.text_area(
            "Paste Instagram caption here:",
            placeholder="Example: Just had the most amazing day ! ",
            height=100
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if caption.strip():
                if model and vectorizer:
                    sentiment = predict_sentiment_custom(caption, model, vectorizer, stop_words)
                    st.markdown(create_sentiment_card(caption, sentiment), unsafe_allow_html=True)
                else:
                    sentiment, score = predict_sentiment_textblob(caption)
                    st.markdown(create_sentiment_card(caption, sentiment, score), unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze!")
    
    elif option == "Multiple Posts Analysis":
        st.markdown("### 📚 Multiple Posts Analysis")
        st.info("Enter each Instagram caption on a new line")
        
        captions_text = st.text_area(
            "Paste multiple captions (one per line):",
            placeholder="""Amazing sunset today! 
Feeling grateful for this moment
Not the best day but tomorrow will be better
Excited for the weekend! """,
            height=200
        )
        
        if st.button("🔍 Analyze All Posts", type="primary"):
            if captions_text.strip():
                captions = [line.strip() for line in captions_text.split('\n') if line.strip()]
                
                if captions:
                    results = analyze_multiple_posts(captions, model, vectorizer, stop_words)
                    
                    # Display results
                    st.markdown("###  Analysis Results")
                    
                    # Summary statistics
                    positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
                    negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
                    neutral_count = sum(1 for r in results if r['sentiment'] == 'Neutral')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(" Positive", positive_count)
                    with col2:
                        st.metric(" Negative", negative_count)
                    with col3:
                        st.metric(" Neutral", neutral_count)
                    
                    st.markdown("---")
                    
                    # Individual results
                    for result in results:
                        st.markdown(
                            create_sentiment_card(
                                result['text'], 
                                result['sentiment'], 
                                result['score'], 
                                result['post_number']
                            ), 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Please enter at least one caption!")
            else:
                st.warning("Please enter some text to analyze!")
    
    elif option == "Bulk Text Analysis":
        st.markdown("###  Bulk Text Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a text file with captions (one per line)",
            type=['txt']
        )
        
        if uploaded_file is not None:
            content = str(uploaded_file.read(), "utf-8")
            captions = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"Loaded {len(captions)} captions from file")
            
            if st.button("🔍 Analyze File Content", type="primary"):
                results = analyze_multiple_posts(captions, model, vectorizer, stop_words)
                
                # Summary
                positive_count = sum(1 for r in results if r['sentiment'] == 'Positive')
                negative_count = sum(1 for r in results if r['sentiment'] == 'Negative')
                neutral_count = sum(1 for r in results if r['sentiment'] == 'Neutral')
                
                st.markdown("###  Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(" Positive", f"{positive_count} ({positive_count/len(results)*100:.1f}%)")
                with col2:
                    st.metric(" Negative", f"{negative_count} ({negative_count/len(results)*100:.1f}%)")
                with col3:
                    st.metric(" Neutral", f"{neutral_count} ({neutral_count/len(results)*100:.1f}%)")
                
                # Download results
                if st.button(" Download Results"):
                    import pandas as pd
                    df = pd.DataFrame(results)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )

    # Footer
    st.markdown("---")
    st.markdown(" **Tip:** For best results, copy full captions including emojis and hashtags!")

if __name__ == "__main__":
    main()
