from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import TwitterCredentials
import re
import json
import csv
import preprocessor as p #not available on pip, so had to manually download


class TwitterAuthenticator:
    '''
    Authenticate using Twitter credentials.
    '''
    def authenticate_twitter_app(self):
        auth = OAuthHandler(TwitterCredentials.ConsumerKey, TwitterCredentials.ConsumerSecret)
        auth.set_access_token(TwitterCredentials.AccessToken, TwitterCredentials.AccessTokenSecret)

        return auth


class TwitterStreamer:
    """
    A class for streaming and processing live tweets.
    """

    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, location, fetch_amount):
        """
        This handles Twitter authentication and the connection to the Twitter Streaming API.
        """
        listener = TwitterListener(fetched_tweets_filename, fetch_amount)
        auth = self.twitter_authenticator.authenticate_twitter_app()

        stream = Stream(auth, listener, tweet_mode='extended')

        #Filter Twitter Streams to capture data by location and language:
        stream.filter(locations=location, languages=['en'])


class TwitterListener(StreamListener):
    """
    This is a basic listener class that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename, fetch_amount):
        self.fetched_tweets_filename = fetched_tweets_filename
        self.counter = 0
        self.fetch_amount = fetch_amount

    def clean_tweet(self, tweet):
        """
        Uses tweet-preprocessor library to clean tweets if wanted.
        """
        p.set_options(p.OPT.URL)
        cleaned_tweet = p.clean(tweet)
        return cleaned_tweet
        #return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split())

    def on_data(self, raw_data):

        try:
            #Parse the tweet using json parser.
            decoded_tweet = json.loads(raw_data)

            #Extract full text from the extended tweet if it exists, else get the short text.
            if ('extended_tweet' in decoded_tweet) and ('full_text' in decoded_tweet['extended_tweet']):
                tweet_text = decoded_tweet['extended_tweet']['full_text']
            elif 'text' in decoded_tweet:
                tweet_text = decoded_tweet['text']

            normal_tweet = decoded_tweet['text']

            #Extract location from decoded tweet.
            tweet_location = decoded_tweet['place']['full_name']

            #Set counter for each write to file.
            i = self.counter
            with open(fetched_tweets_filename, 'a', encoding = 'utf-8', newline='') as tf:
                '''
                Remove replies, and any tweets that contain links because 
                extended tweets that contain links also become truncated.
                
                Short, non-truncated tweets that contain links are media links, 
                so those will also be removed as a result.
                
                All resulting tweets in the file are non-truncated tweets with no media attached.
                Retweets are also removed.
                '''
                if not (tweet_text.startswith('@')
                        or re.compile("http\S+").search(tweet_text))\
                        or re.compile("RT @").search(tweet_text):

                    #Terminate process if counter reaches limit.
                    self.counter += 1
                    if self.counter > self.fetch_amount:
                        return False

                    #tweet_text = self.clean_tweet(tweet_text)

                    #Remove line breaks.
                    tweet_text = tweet_text.replace('\n',' ')

                    csv_writer = csv.writer(tf)
                    csv_writer.writerow([tweet_text, tweet_location])

                    print([normal_tweet, tweet_text, i])
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status_code):
        # stop stream if rate limit warning received
        if status_code == 420:
            return False
        print(status_code)


if __name__ == '__main__':

    twitter_streamer = TwitterStreamer()

    London = [-0.510375, 51.28676, 0.334015, 51.691874]
    Leeds = [-1.800421, 53.698967, -1.290352, 53.945871]
    London_or_Leeds = [-0.510375, 51.28676, 0.334015, 51.691874, -1.800421, 53.698967, -1.290352, 53.945871]
    NewYork = [-74.25909,40.477399,-73.700181,40.916178]

    fetched_tweets_filename = 'tester' #"tweets_London_230619_2AM.csv"

    location = London

    fetch_amount = 10000

    twitter_streamer.stream_tweets(fetched_tweets_filename, location, fetch_amount)



