import numpy as np
from nltk.corpus import twitter_samples

def tweet_loader(sentiment = None):
    """
    Function that loads the twitter samples.

    Args:
        sentiment (string): Optional. If no sentiment is given, all positive and negative tweets are returned.
        If "positive" is passed, only the positive tweets are returned.
        If "negative" is passed, only the negative tweets are returned.

    Returns:
        tweets (List): List with all tweets
    
    """

    if sentiment == "positive":
        tweets = twitter_samples.strings('positive_tweets.json')
        labels = np.ones((len(all_positive_tweets)))
    elif sentiment == "negative":
        tweets = twitter_samples.strings('negative_tweets.json')
        labels = np.zeros((len(all_negative_tweets)))
    else:
        # select the lists of positive and negative tweets
        all_positive_tweets = twitter_samples.strings('positive_tweets.json')
        all_negative_tweets = twitter_samples.strings('negative_tweets.json')

        # concatenate the lists, 1st part is the positive tweets followed by the negative
        tweets = all_positive_tweets + all_negative_tweets

        # make a numpy array representing labels of the tweets
        labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

    return tweets, labels