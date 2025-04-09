#### StopWord Removal

import pandas as pd
import ollama
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv('Data/uk-ge-2019-sample_processed.csv')


df3 = []

for index, row in df.iterrows():
    if '@' in row['text']:
        df3.append(row.to_list())
        if len(df3) == 3:
                break
        

df3 = pd.DataFrame(df3, columns = df.columns)

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


df3['cleaned_tweets']=  remove_stopwords_and_punctuation(df3['text'])


superNarratives = ["Anti-EU-related narratives","Anti-Elites","Distrust in democratic system-related narratives","Distrust in institutions-related narratives","Ethnic-related", "Gender-related", "Geopolitics", "Migration-related", "Political hate-related", "Religious-related"]

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


response = []

for i in range(0, len(df3)):
    response_super = []
    for j in range(0,len(superNarratives)):
        prompt = prompt_generator(df3,i,superNarratives[j],'cleaned_tweets')
        response_super.append(ollama.generate(model='llama2', prompt=prompt)['response'])
    response.append(response_super)



response2 = []

for i in range(0, len(df3)):
    response_super = []
    for j in range(0,len(superNarratives)):
        prompt = prompt_generator(df3,i,superNarratives[j],'cleaned_tweets')
        response_super.append(ollama.generate(model='llama2', prompt=prompt)['response'])
    response2.append(response_super)


print(response, response2)