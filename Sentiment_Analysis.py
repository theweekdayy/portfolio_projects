import pandas as pd
from textblob import TextBlob
df = pd.read_csv('GrammarandProductReviews.csv')
olay_df = df[df['brand'] == "Olay"]
olay_df = olay_df.head(600)
olay_df['sentiment'] = olay_df['reviews.text'].apply(lambda x: TextBlob(x).sentiment.polarity)
def get_sentiment_category(sentiment):
    if sentiment > 0:
        return 'positive'
    elif sentiment == 0:
        return 'neutral'
    else:
        return 'negative'

olay_df['sentiment_category'] = olay_df['sentiment'].apply(get_sentiment_category)
positive_reviews = olay_df[olay_df['sentiment_category'] == 'positive']
neutral_reviews = olay_df[olay_df['sentiment_category'] == 'neutral']
negative_reviews = olay_df[olay_df['sentiment_category'] == 'negative']
print("Number of positive reviews:", len(positive_reviews))
print("Number of neutral reviews:", len(neutral_reviews))
print("Number of negative reviews:", len(negative_reviews))

from collections import Counter
import re
olay_reviews = olay_df['reviews.text'].str.lower().str.cat(sep=' ')
words = re.findall(r'\b\w+\b', olay_reviews)
word_counts = Counter(words)
most_common_words = word_counts.most_common(100)


import nltk
from collections import Counter
import re
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
olay_reviews = olay_df['reviews.text'].str.lower().str.cat(sep=' ')
words = re.findall(r'\b\w+\b', olay_reviews)
words_filtered = [word for word in words if word not in stop_words]
word_counts = Counter(words_filtered)
most_common_words = word_counts.most_common(150)
#print(most_common_words)
most_common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])
print(most_common_words_df[['Word', 'Count']])


import matplotlib.pyplot as plt
most_common_words = word_counts.most_common(20)
labels, values = zip(*most_common_words)
plt.bar(labels, values)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Most Common Words in Text Reviews of Olay')
plt.show()

