import pandas as pd
import ollama

from nltk.corpus import stopwords
import string
import re

def pre_process_pipeline(df_col):
    df_col = remove_mentions(df_col)
    df_col = remove_stopwords_and_punctuation(df_col)
    df_col = tokenize_tweets(df_col)
    return df_col


def remove_stopwords_and_punctuation(tweet_column):
    stop_words = set(stopwords.words('english'))
    cleaned_tweets = []

    for tweet in tweet_column:
        # Remove punctuation
        tweet = tweet.translate(str.maketrans("", "", string.punctuation))

        # Remove stop words
        words = tweet.split()
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
        cleaned_tweets.append(' '.join(filtered_words))

    return cleaned_tweets


def remove_mentions(tweet_column):
    # Use regular expression to remove mentions
    pattern = re.compile(r'@[a-zA-Z0-9_]+')
    cleaned_tweets = [re.sub(pattern, '', tweet) for tweet in tweet_column]
    return cleaned_tweets

def remove_hashtags(tweet_column):
    # Use regular expression to remove mentions
    pattern = re.compile(r'#[a-zA-Z0-9_]+')
    cleaned_tweets = [re.sub(pattern, '', tweet) for tweet in tweet_column]
    return cleaned_tweets

def tokenize_tweets(tweet_column):
    # Split and strip each tweet into tokens
    tokenized_tweets = [tweet.split() for tweet in tweet_column]
    return tokenized_tweets


# Apply the text processing functions to the 'tweets' column



def prompt_generator(df,i,task_name,col_name):
    prompt = """
    **Task:** Determine the relevance of the following text to the label. Please read the text and provide a binary output (1 for relevant, 0 for not relevant) for the given label.

    **Label:**
    """ + str(task_name) + """

    **Text to be categorized:**
    """ + str(df[col_name][i]) + """

    **Output:**
    Please provide ONLY the binary value of 1 or 0 WITHOUT additional explanation. Don't output any words or sentences.
    """
    return prompt

def response_LLM(df,superNarratives,col_name):
    response = []
    for i in range(0, len(df)):
        response_super = []
        for j in range(0,len(superNarratives)):
            prompt = prompt_generator(df,i,superNarratives[j],col_name)
            response_super.append(ollama.generate(model='llama2', prompt=prompt)['response'])
        response.append(response_super)
    return response


