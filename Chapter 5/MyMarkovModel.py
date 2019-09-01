import numpy as np
import pandas as pd
import collections as c
import time
import nltk
import regex as re
from itertools import chain
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # Stop irrelevant warnings from showing.

'''
Run if first time running code to get the corpus required.
'''


# nltk.download('punkt')
# nltk.download('stopwords')


def file_to_DF(tweets_file):  # Add ngram parameter
    '''
    Convert tweet file to a dataframe with Tweet and Class as the column names, and also preprocesses each tweet.
    '''
    tweets_df = pd.read_csv(tweets_file, header=None)
    tweets_df = tweets_df.rename(columns={0: 'Tweet', 1: 'Class'})
    # tweets_df['Preprocessed_Tweet'] = tweets_df['Tweet']
    return tweets_df


class TweetClassifier:

    def run_classifier(self, training_set, test_set, word_or_character, ngram, number_of_words,
                       min_transition_probability, missing_word_penalty):
        # Start timing.
        start_time = time.time()

        training_set['Preprocessed_Tweet'] = training_set.Tweet.apply(
            (lambda tweet: self.tweet_preprocessor(tweet, word_or_character)))

        training_set = training_set[training_set.Preprocessed_Tweet.astype(str) != '[]']

        test_set['Preprocessed_Tweet'] = test_set.Tweet.apply(
            (lambda tweet: self.tweet_preprocessor(tweet, word_or_character)))

        test_set = test_set[test_set.Preprocessed_Tweet.astype(str) != '[]']

        tweet_classifier = TweetClassifier()

        tweet_classes = tweet_classifier.get_classes(training_set)

        classes_DF = tweet_classifier.split_DF(training_set, tweet_classes)

        class_transition_matrices = tweet_classifier.class_transition_matrices(classes_DF, ngram, number_of_words)

        class_initial_states = tweet_classifier.class_initial_probabilities(classes_DF, ngram)

        classified_df = tweet_classifier.classify_corpus(test_set, ngram, class_initial_states,
                                                         class_transition_matrices, min_transition_probability,
                                                         missing_word_penalty)

        results = tweet_classifier.evaluation(classified_df)

        run_time = round(time.time() - start_time, 2)

        results = {
            'Tokenise by: {}, Number of tokens: {}, Ngrams: {}, Min. trans. prob.: {}, Missing state penalty: {}'.format(
                word_or_character, number_of_words, ngram, min_transition_probability, missing_word_penalty): results}
        results = pd.DataFrame.from_dict(results)
        results.loc['Run_Time'] = run_time
        return results

    def evaluation(self, classified_df):
        results = {}
        classes = classified_df.Class.unique()
        total_rows = len(classified_df.index)
        accurate_predictions = classified_df[classified_df['Class'] == classified_df['Predicted']].count()
        accuracy = accurate_predictions / total_rows
        results['Accuracy'] = accuracy[1]

        for c in classes:
            true_positives = classified_df.loc[
                (classified_df['Predicted'] == c) & (classified_df['Class'] == c)].count()
            false_positives = classified_df.loc[
                (classified_df['Predicted'] == c) & (classified_df['Class'] != c)].count()
            false_negatives = classified_df.loc[
                (classified_df['Predicted'] != c) & (classified_df['Class'] == c)].count()
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            results['{}_Precision'.format(c)] = precision[1]
            results['{}_Recall'.format(c)] = recall[1]

        return results

    def tweets_to_DF(self, tweets_file, word_or_character):  # Add ngram parameter
        '''
        Convert tweet file to a dataframe with Tweet and Class as the column names, and also preprocesses each tweet.
        '''
        tweets_df = pd.read_csv(tweets_file, header=None)
        tweets_df = tweets_df.rename(columns={0: 'Tweet', 1: 'Class'})
        # tweets_df['Preprocessed_Tweet'] = tweets_df['Tweet']

        tweets_df['Preprocessed_Tweet'] = tweets_df.Tweet.apply(
            (lambda tweet: self.tweet_preprocessor(tweet, word_or_character)))

        tweets_df = tweets_df[tweets_df.Preprocessed_Tweet.astype(str) != '[]']

        return tweets_df

    # def training_test_split(self, tweets_df, percentage_split):

    def most_common_words(self, tweets_df, number_words=None):
        tweets_list = list(tweets_df.Preprocessed_Tweet)
        flatten_tweets_list = list(chain.from_iterable(tweets_list))

        word_counter = c.Counter(flatten_tweets_list)
        most_common_list = word_counter.most_common(number_words)

        return most_common_list

    def tweet_preprocessor(self, tweet, word_or_character):  # add most common words filter
        if word_or_character == 'word':
            stopwords = set(nltk.corpus.stopwords.words('english'))

            tweet = tweet.lower()
            tweet = re.sub(r'\d+', '', tweet)
            tweet = re.sub("[^\P{P}#@&]+", "", tweet)
            tweet = re.sub('&amp', '&', tweet)
            tweet = re.sub('&lt', '<', tweet)
            tweet = re.sub('&gt', '>', tweet)
            tweet = re.sub("[^\P{P}@#]+", "", tweet)

            whitespace_tokenize = nltk.WhitespaceTokenizer()
            tweet = whitespace_tokenize.tokenize(tweet)

            tweet = [word for word in tweet if word not in stopwords]

            stemmer = nltk.stem.PorterStemmer()
            tweet = [stemmer.stem(word) for word in tweet]

            tweet.append('◻')

            anonymise_users = []
            for word in tweet:
                if word.startswith('@'):
                    word = '@USER'
                else:
                    pass
                anonymise_users.append(word)

            final_tweet = anonymise_users
            return final_tweet

        elif word_or_character == 'character':
            tweet = tweet.lower()
            tweet = re.sub('&amp;', '&', tweet)
            tweet = re.sub('&lt;', '<', tweet)
            tweet = re.sub('&gt;', '>', tweet)
            tweet = list(tweet)
            tweet.append('◻')
            return tweet

    def get_classes(self, tweets_df):
        '''
        Get unique classes from dataframe.
        '''
        tweet_classes = tweets_df.Class.unique()
        return tweet_classes

    def split_DF(self, tweets_df, tweet_classes):
        '''
        Split the dataframe by class and put the new dataframes into a dictionary.
        '''
        split_DF = {}
        for i in range(len(tweet_classes)):
            class_name = tweet_classes[i]

            # Get dataframe for each class.
            class_df = tweets_df[tweets_df.Class == class_name]

            # Store in dictionary with class name as key.
            split_DF['{0}'.format(class_name)] = class_df
        return split_DF

    def class_transition_matrices(self, split_DF, ngram, number_of_words):
        '''
        Get transition matrices for each class in a tweet dataframe.
        ngram is the order of the Markov model.
        '''
        markov_model = MarkovModel()

        classes = list(split_DF.keys())

        # Create dictionary to store transition matrices for each class.
        class_transition_matrices = {}

        for i in classes:
            class_name = i

            # Get dataframe for each class.
            class_df = split_DF[class_name]

            # Calculate transition matrix and store in dictionary with class name as key.
            class_transition_matrices['{0}'.format(class_name)] = markov_model.get_transition_matrix(class_df, ngram,
                                                                                                     number_of_words)

        return class_transition_matrices

    def class_initial_probabilities(self, split_DF, ngram):
        '''
        Get initial state vectors for each class.
        '''
        markov_model = MarkovModel()

        # Extract class names
        classes = list(split_DF.keys())

        # Create dictionary to store initial state vectors for each class.
        class_initial_probabilities = {}

        for c in classes:
            class_name = c

            # Get dataframe for each class.
            class_df = split_DF[class_name]

            # Calculate initial state vectors and store in dictionary with class name as the key.
            class_initial_probabilities['{0}'.format(class_name)] = markov_model.get_initial_state_vector(class_df,
                                                                                                          ngram)
        return class_initial_probabilities

    def loglik_list(self, tweet, ngram, class_initial_probabilities, class_transition_matrices,
                    min_transition_probability, missing_word_penalty):
        '''
        Takes a tweet and returns the log-likelihood of each class trained on a tweet corpus.
        '''
        markov_model = MarkovModel()

        # Get class names.
        classes = list(class_initial_probabilities.keys())

        # Initialise list to store log-likelihood values.
        loglik_list = []

        for i in range(len(classes)):
            # Iterate over each class transition matrix and initial state probabilities.
            transition_matrix = class_transition_matrices[classes[i]]
            initial_probability = class_initial_probabilities[classes[i]]

            # Calculate log-likelihood.
            loglik = markov_model.get_loglikelihood(tweet, ngram, initial_probability, transition_matrix,
                                                    min_transition_probability, missing_word_penalty)

            # Append to list.
            loglik_list.append([loglik, classes[i]])

        return loglik_list

    def classify_tweet(self, tweet, ngram, class_initial_probabilities, class_transition_matrices,
                       min_transition_probability, missing_word_penalty):
        '''
        Returns a predicted class for a tweet.
        '''
        loglik_list = self.loglik_list(tweet, ngram, class_initial_probabilities, class_transition_matrices,
                                       min_transition_probability, missing_word_penalty)
        maximum_loglik_class = max(loglik_list)

        return maximum_loglik_class[1]

    def classify_corpus(self, test_set, ngram, class_initial_probabilities, class_transition_matrices,
                        min_transition_probability, missing_word_penalty):
        '''
        Classifies a test set using a training set to train the classifier.
        '''
        test_set['Predicted'] = test_set.Preprocessed_Tweet.apply(
            lambda tweet: TweetClassifier.classify_tweet(self, tweet, ngram, class_initial_probabilities,
                                                         class_transition_matrices, min_transition_probability,
                                                         missing_word_penalty))

        classified_df = test_set[['Tweet', 'Class', 'Predicted']]

        return classified_df


class MarkovModel:

    def get_initial_state_vector(self, tweets_df, ngram):
        # Tokenise each row of tweets and store them in a list.
        tweets_list = tweets_df['Preprocessed_Tweet'].tolist()

        # Take the first word of each tweet stored in the tokenised tweets list.
        first_word_list = [tuple(tweet[0:ngram]) for tweet in tweets_list]

        # Count the number of first words that occur.
        first_word_count = c.Counter(first_word_list)
        initial_probabilities = c.defaultdict()

        for initial_state in first_word_count.keys():
            total = sum(first_word_count.values())
            count = list(first_word_count.values())
            initial_probabilities[initial_state] = count[
                                                       0] / total  # for initial_state, count in first_word_count.items()}

        return initial_probabilities

    def get_transition_matrix(self, tweets_df, ngram, number_words_filter):
        # Get the most common words and their counts from the Preprocessed_Tweets.
        most_common_words = TweetClassifier().most_common_words(tweets_df, number_words_filter)

        # Remove the word counts from the most common list.
        most_common_words_list = [word[0] for word in most_common_words]

        # Initialise dictionary with counter.
        tweet_model = c.defaultdict(c.Counter)

        # Store preprocessed tweets as a list.
        tweets_list = tweets_df.Preprocessed_Tweet.tolist()

        # Only keep most common words.
        tweets_list = [[word for word in tweet if word in most_common_words_list] for tweet in tweets_list]

        # Remove tweets that are too small for the order of the Markov model.
        tweets_list = [tweet for tweet in tweets_list if len(tweet) >= 2 * ngram]

        # Remove empty sublists.
        tweets_list = [tweet for tweet in tweets_list if tweet]

        # Iterate over each state and its next state and count the number of occurrences.

        for tweet in tweets_list:
            possible_states = range(len(tweet) - ngram)
            for i in possible_states:
                state = tuple(tweet[i*ngram: i*ngram + ngram])
                next_state = tuple(tweet[i*ngram + ngram: i*ngram + 2 * ngram])

                # Update state key with the count value of the next state.
                tweet_model[state][next_state] += 1

        probabilities = {}
        for state, next_state in tweet_model.items():
            total = sum(next_state.values())
            probabilities[state] = {next_state: count / total for next_state, count in next_state.items()}

        '''
        state_list = [state for state in probabilities]

        next_state_list = []
        for next_state in probabilities.values():
            next_state_list.extend(list(next_state.keys()))


        probability_list = []
        for next_state in probabilities.values():
            probability_list.extend(list(next_state.values()))

        #model_df = pd.DataFrame({'State':state_list, 'Next_State':next_state_list,'Transition_Probability':probability_list})

        # Convert dictionary into a model and replace NaN values with 0.
        #transition_matrix = pd.DataFrame(model_df).fillna(0)
        '''

        return probabilities

    def log0(self, x, min_transition_probability):
        '''
        Return the natural log of values above 0, else return 0 to avoid infinity errors since log(0) is undefined.
        '''
        return np.log(min_transition_probability) if x <= 0 else np.log(x)

    def get_loglikelihood(self, tweet, ngram, initial_state_vector, transition_matrix, min_transition_probability,
                          missing_word_penalty):
        '''

        '''

        # Extract first word of tweet.
        first_word = tuple(tweet[0:ngram])
        # print('first word:',first_word)
        # print('test2',initial_state_vector[first_word])
        # Initialise list to store likelihood values.
        likelihood_list = []

        missing_words_count = 0

        # Try getting probability of first word of tweet
        try:
            # Extract the probability value of the first word of the tweet in the initial state vector.
            first_word_probability = initial_state_vector[first_word]
            # print('first word probability:', first_word_probability)
            # first_word_probability = list(first_word_probability.values())
            # first_word_probability = first_word_probability[0]
            # print('Try first word:',first_word)
            # print('First word probability:',first_word_probability)
            # Append to likelihood list.
            likelihood_list.append(first_word_probability)


        except:
            # Check whether initial state exists in transition matrix.
            if first_word in transition_matrix.keys():
                likelihood_list.append(0)
                # print('Does not exist in initial state vector but exists in transition matrix - set first word probability to 0.')

            else:
                missing_words_count = missing_words_count + 1
                # print('Does not exist in initial state vector or transition matrix')

        possible_states = range(len(tweet) - ngram)
        # Iterate for each state (word) and next state (next word) in tweet.
        for i in possible_states:
            state = tuple(tweet[i: i + ngram])
            next_state = tuple(tweet[i + ngram: i + 2 * ngram])
            # print('state:',state)
            # print('next state:',next_state)
            # Try getting transition probability of each state into next state.

            '''
            current_state = transition_matrix.get(state, None)
            print('current_state: ',current_state)
            #print('transition probability:', transition_probability)

            if current_state is None:
                print('no transition probability')
                missing_words_count = missing_words_count + 1
            else:
                next_state2 = current_state.get(current_state).get(next_state)
                print('next_state2:',next_state2)
                likelihood_list.append(next_state2)
            '''

            try:
                transition_probability = transition_matrix[state][next_state]
                # print('Try state: ',state)
                # print('Try next state:',next_state)
                # print('Try transition probability:',transition_matrix[state][next_state])
                likelihood_list.append(transition_probability)
            except:
                # Pass on error if word doesn't exist in row labels for transition matrix.
                # print('No transition probability.')
                missing_words_count = missing_words_count + 1

        # print(likelihood_list)

        # List comprehension for log-likelihood.
        loglik_list = [self.log0(i, min_transition_probability) for i in likelihood_list]
        # print('Probabilities:',likelihood_list)
        # print('Log-likelihood list:', loglik_list)
        # print('Number of missing states:', missing_words_count)
        # print('Missing state penalty set to:', missing_word_penalty)
        # Calculate final log-likelihood value.
        loglik = sum(loglik_list) + missing_words_count * missing_word_penalty
        return loglik


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    '''
    -----------------------------------LOAD FILES ------------------------------------------
    '''

    file = 'final_dataset.csv'
    df = file_to_DF(file)
    y = df.Class
    training_set, test_set = train_test_split(df, test_size=0.5, stratify=y)

    tweet_classifier = TweetClassifier()
    markov_model = MarkovModel()

    '''
    -----------------------------------PARAMETERS ------------------------------------------
    '''

    word_or_character = 'character'
    ngram = 3
    number_of_words = None
    min_transition_probability = 0.001
    missing_word_penalty = -7


    '''
    ---------------------------------------CLASSIFY---------------------------------------
    '''


    run = tweet_classifier.run_classifier(training_set, test_set, word_or_character, ngram, number_of_words,
                                          min_transition_probability, missing_word_penalty)

    print(run)


    '''
    -----------------------------------EXAMPLE--------------------------
    '''

    '''
    training_file = 'example_training.csv'
    training_set = file_to_DF(training_file)

    test_file = 'example_test.csv'
    test_set = file_to_DF(test_file)

    word_or_character = 'character'
    training_set['Preprocessed_Tweet'] = training_set.Tweet.apply((lambda tweet: tweet_classifier.tweet_preprocessor(tweet, word_or_character)))

    training_set = training_set[training_set.Preprocessed_Tweet.astype(str) != '[]']

    test_set['Preprocessed_Tweet'] = test_set.Tweet.apply((lambda tweet: tweet_classifier.tweet_preprocessor(tweet, word_or_character)))

    test_set = test_set[test_set.Preprocessed_Tweet.astype(str) != '[]']


    ngram = 1
    number_of_words = None
    min_transition_probability = 0.001
    missing_word_penalty = -5

    tweet = 'ababab◻'
    classes = tweet_classifier.get_classes(training_set)
    class_df = tweet_classifier.split_DF(training_set, classes)
    class_initial_state_prob = tweet_classifier.class_initial_probabilities(class_df,ngram)
    class_tmatrices = tweet_classifier.class_transition_matrices(class_df, ngram, number_of_words)
    loglik_list = tweet_classifier.loglik_list(tweet, ngram, class_initial_state_prob, class_tmatrices, min_transition_probability, missing_word_penalty)
    prediction = tweet_classifier.classify_tweet(tweet, ngram, class_initial_state_prob, class_tmatrices, min_transition_probability, missing_word_penalty)
    classify_corpus = tweet_classifier.classify_corpus(test_set,ngram,class_initial_state_prob,class_tmatrices,min_transition_probability,missing_word_penalty)
    print('Leeds Initial State Probabilities: ', dict(class_initial_state_prob['Leeds']))
    print('Leeds Transition Matrix: ', dict(class_tmatrices['Leeds']))
    print('London Initial State Probabilities: ', dict(class_initial_state_prob['London']))
    print('London Transition Matrix: ', dict(class_tmatrices['London']))
    print('Log-Likelihood List: ', loglik_list)
    print('Predicted Class: ', prediction)
    '''

    #DICT = dict(class_initial_state_prob['London'])
    #print(DICT[('a',)])

