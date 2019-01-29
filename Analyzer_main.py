from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

#Function to extract features

def extract_features(words):
    return dict([(word, True) for word in words])

if __name__ == '__main__':
    #Loading reviews from corpus
    fileids_pos = movie_reviews.fileids('pos')
    fileids_neg = movie_reviews.fileids('neg')

    #Extracting features from reviews
    features_pos = [(extract_features(movie_reviews.words(fileids = [f])), 'Positive') for f in fileids_pos]
    features_neg = [(extract_features(movie_reviews.words(fileids = [f])), 'Negative') for f in fileids_neg]


    #Defining test and train split
    #80% for training and 20% for testing
    threshold = 0.8
    num_pos = int(threshold * len(features_pos))
    num_neg = int(threshold * len(features_neg))

    #Creating training and testing datasets
    features_train = features_pos[:num_pos] + features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[num_neg:]


    #Number of datapoints
    print('\nNumber of training datapoints:', len(features_train))
    print('\nNUmber of test datapoints:', len(features_test))

    #CLassifeir training
    classifier = NaiveBayesClassifier.train(features_train)
    print('\nAccuracy of the classifier:', nltk_accuracy(classifier, features_test))

    #Most informative words (15)
    N = 15
    print('\nTop' + str(N) + 'most informative words:')
    for i, item in enumerate(classifier.most_informative_features()):
        print(str(i+1) + '.' + item[0])
        if i == N-1:
            break

    #Testing of input reviews
    input_reviews = [
        'A must watch amazingly directed movie. I would recommend this to everyone',
        'Costumes and apparels used by the cast were great',
        'Directed by a famous director who is well known for his amazing work',
        'The story made no sense. I would classify it as an idiotic movie',
    ]

    #Output prediction

    print('\n MOvie review predictions')
    for review in input_reviews:
        print('\nReview', review)
        probabilities = classifier.prob_classify(extract_features(review.split()))
        predicted_sentiment = probabilities.max()
        print("Predicted sentiment:", predicted_sentiment)
        print("Probability:", round(probabilities.prob(predicted_sentiment), 2))
