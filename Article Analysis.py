# Updated imports with additional libraries for styling
import streamlit as st
import newspaper
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from nltk import ngrams

# Function to extract article content from a given URL
def extract_article_content(url):
    article = Article(url, language="en")
    article.download()
    article.parse()
    article.nlp()
    return {
        'URL': url,
        'Topic': article.title,
        'Text': article.text,
        'Summary': article.summary,
        'Keywords': ', '.join(article.keywords)
    }

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

def generate_word_cloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, format='png')  # Save the word cloud as a PNG file
    plt.close()

# Custom function for styling the Streamlit app
def set_custom_style():
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url("https://images.unsplash.com/photo-1699009436134-b87643babed9?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8MTR8fHxlbnwwfHx8fHw=");
                background-size: cover;
                background-repeat: no-repeat;
            }}
            .st-d6 {{
                background-color: #f0f8ff;  /* Light Blue */
            }}
            .st-bb {{
                background-color: #4169e1;  /* Royal Blue */
                color: #ffffff;  /* White */
            }}
            .st-cg {{
                color: #20b2aa;  /* Light Sea Green */
            }}
            .st-eq {{
                background-color: #f5f5f5;  /* Silver */
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit web application
def main():
    set_custom_style()
    st.title("Article Analysis")
    
    # Get user input for the URL
    url = st.text_input("Enter the article URL:")
    
    if st.button("Extract Data"):
        # Extract article content
        article_data = extract_article_content(url)
        
        # Create a dataframe from the extracted data
        df = pd.DataFrame([article_data])
        
        # Display the dataframe
        st.subheader("Article Data:")
        st.dataframe(df.style.set_properties(**{'background-color': '#f0f8ff', 'color': 'black'}))
        
        # Perform sentiment analysis
        sentiment_score = perform_sentiment_analysis(article_data['Text'])
        st.subheader("Sentiment Analysis:")
        st.write("Sentiment Score:", sentiment_score)
        
        # Generate word cloud
        unigrams = article_data['Text'].split()
        bigrams = [' '.join(grams) for grams in ngrams(article_data['Text'].split(), 2)]
        
        st.subheader("Word Clouds:")
        
        # Unigram Word Cloud
        st.subheader("Unigram Word Cloud:")
        generate_word_cloud(' '.join(unigrams), "Unigram - " + article_data['Topic'], "unigram_wordcloud.png")
        st.image("unigram_wordcloud.png", width=800)  # Adjust width as needed

        # Bigram Word Cloud
        st.subheader("Bigram Word Cloud:")
        generate_word_cloud(' '.join(bigrams), "Bigram - " + article_data['Topic'], "bigram_wordcloud.png")
        st.image("bigram_wordcloud.png", width=800)  # Adjust width as needed

# Run the Streamlit application
if __name__ == '__main__':
    main()
