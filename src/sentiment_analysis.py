import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/Purbasha/Downloads/sentimentdataset.csv")

# Show columns to confirm
print("Columns in dataset:", df.columns)

# Show first few rows
print("Sample Data:")
print(df.head())

# Use the correct column name 'Text' (case-sensitive)
text_col = 'Text'

# Define function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Apply sentiment analysis
df['Sentiment'] = df[text_col].apply(get_sentiment)

# Add sentiment label
df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Count sentiment types
sentiment_counts = df['Sentiment_Label'].value_counts()

# Show sentiment distribution
print("\nSentiment Distribution:")
print(sentiment_counts)

# Visualization
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Analysis Result')
plt.xlabel('Sentiment')
plt.ylabel('Number of Posts')
plt.tight_layout()

# Save plot - adjust folder if needed
plt.savefig('sentiment_plot.png')
plt.show()
