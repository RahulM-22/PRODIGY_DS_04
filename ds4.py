import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import re

# Load your dataset
df = pd.read_csv('D:/MP 1 FINAL/DS 04/twitter_training.csv/twitter_training.csv', header=None)

# Assign column names based on the structure
df.columns = ['id', 'entity', 'sentiment', 'text']

# Data Cleaning Function
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#', '', text)  # Remove hashtag symbols
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    return ''  # Return an empty string if the text is not valid

# Clean the text
df['cleaned_text'] = df['text'].apply(clean_text)

# Sentiment Analysis using TextBlob
def get_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Plot Sentiment Distribution
sns.countplot(x='sentiment', data=df, palette="viridis")
plt.title('Sentiment Distribution')
plt.show()

# Word Cloud for Positive Sentiment
positive_words = ' '.join([text for text in df[df['sentiment'] == 'Positive']['cleaned_text']])
wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')
plt.show()
