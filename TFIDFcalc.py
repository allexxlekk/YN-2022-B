"""
This file contains functions that calculate the mean TF-IDF value for each word in the vocabulary
from the data files. Also contains a function for saving/loading calculated values to/from a .dat file.
"""
import numpy as np
from sklearn import preprocessing
from collections import defaultdict


def parse_file(filename):
    """Parses selected file and returns a list of articles containg their words."""
    articles_list = []

    with open(filename, "r") as f:
        # Parse each article.
        for line in f:
            # Extract words from article.
            articles_list.append([int(s) for s in line.split() if s.isdigit()])
    return articles_list


def get_article_data(articles):
    """Gets data from the list of articles and stores them in a dictionary."""
    word_dictionary = defaultdict(list)  # {key : []}

    # Each word is a key for the dictionary.
    # The value of each key is a list that contains the Term Frequency value for each article.
    # The length of the list is the Document Frequency.
    # This method works because we care about the average TF-IDF valuey
    # and which article contains each word is irrelevant information.

    for a in articles:
        # Check each word one time for each article.
        unique_words = list(set(a))
        for w in unique_words:
            word_count = a.count(w)  # Get the TF value.
            word_dictionary[w].append(word_count)
    return word_dictionary


def TFIDF(word_value, number_of_articles):
    """Calculates the mean TF-IDF score for a word."""
    TFIDF_list = []
    # Compute the IDF value of word.
    DF = len(word_value)
    IDF = np.log(number_of_articles / DF)
    # Compute every TF-IDF value for word.
    for TF in word_value:
        TFIDF_list.append(TF * IDF)

    return np.mean(TFIDF_list)


def TFIDFdict(word_dict, no_articles):
    """Stores the mean TF-IDF value of each word in a dictionary."""

    # Array for normalizing TF-IDF values.
    norm_array = []

    # Calculate TF-IDF value for each word and save it to an array.
    for _, value in sorted(word_dict.items()):
        norm_array.append(TFIDF(value, no_articles))

    # Normalize array.
    norm_array = preprocessing.normalize([np.array(norm_array)])[0]

    # Save to a dictionary.
    for i, TFIDF_score in enumerate(norm_array):
        word_dict[i] = TFIDF_score

    return word_dict


def parse(filename):
    """Returns a dictionary with word,mean(TF-IDF) key-value pair for each word in the file."""
    articles = parse_file(filename)
    dikt = get_article_data(articles)
    return TFIDFdict(dikt, len(articles))


def save(filename):
    """Saves TF_DF value for each word in a .dat file."""

    final_dict = parse("combined.dat")  # Create dictionary.

    with open(filename, "w") as f:  # Save to file in "word" : "value" format.
        for i in range(0, len(final_dict)):
            s = str(i) + " : " + str(final_dict[i])
            f.write(s + "\n")


def load(filename):
    """Loads values from the specified file to a list and returns it."""

    load_list = []

    with open(filename, "r") as f:
        for line in f:
            v = line.split(":")  # v[0] = "word", v[1] = "tfidf value"
            load_list.append(float(v[1].strip()))

    return load_list
