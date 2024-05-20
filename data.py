# %%
import nltk
from nltk.corpus import stopwords
import re
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pickle

# %% [markdown]
# 

# %%
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))

# Read data
data = pd.read_csv("FinalBalancedDataset.csv")
data.tail(10)

# %%
# Map class labels to more descriptive labels
data["labels"] = data["Toxicity"].map({0: "No Hate and Offensive Speech", 1: "Offensive Language"})
data = data[["tweet", "labels"]]

# %%
data

# %%
def clean(text):
    text = str(text).lower()
    text = re.sub('^.*?:', '', text)  # Remove all words before and including the first colon
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply the clean function to the tweet column
data["tweet"] = data["tweet"].apply(clean)

# %%
x = np.array(data["tweet"])
y = np.array(data["labels"])
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train


# %%
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# %%
print("Model Accuracy:", model.score(x_test, y_test))

# %%
sample = "I love my mom"
data = cv.transform([sample]).toarray()
print(model.predict(data))

# %%
pickle.dump(model, open("hatehmodel.pkl", 'wb'))
pickle.dump(cv, open("cv.pkl", 'wb'))

# %%
def generate_wordcloud_from_csv(file_path):
    df = pd.read_csv('FinalBalancedDataset.csv')
    
    # Clean text: remove all words before the colon and apply the clean function
    df['cleaned_tweet'] = df['tweet'].apply(lambda x: clean(x.split(':', 1)[-1] if ':' in x else x))
    text = " ".join(tweet for tweet in df['cleaned_tweet'])
    
    stopwords = set(STOPWORDS)
    stopwords.update(["is", "in", "and", "RT", "then","user","ð"])
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='Reds',
        stopwords=stopwords,
        max_font_size=100,
        random_state=42
    ).generate(text)

    return wordcloud

wordcloud = generate_wordcloud_from_csv("FinalBalancedDataset.csv")

# %%
# Display the generated Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%

# %%


# %%


# %%


# %%


# %%



