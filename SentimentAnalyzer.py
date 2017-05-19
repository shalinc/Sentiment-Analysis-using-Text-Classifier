# coding: utf-8

"""
	Sentiment Analyzer using Text Classifier
"""

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import urllib.request


def read_data(path):
    """
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    """
    ###TODO

    # We don't need any punctuations here, all internal punctuation must be removed too
    if keep_internal_punct is False:
        return np.asarray(re.sub("\W+",' ',doc.lower()).split())
    # We need to keep internal punctuation and remove (strip) leading or trailing punctuations
    elif keep_internal_punct is True:
        #return (np.asarray(re.sub("[^\w']+",' ',doc).lower().split()))
        return np.asarray([token.strip(string.punctuation) for token in doc.lower().split()])

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    """
    ###TODO

    # Count the no. of occurrences for a token, update the feats dictionary
    for token in tokens:
        feats["token="+token] += 1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    
    """
    ###TODO

    # First make windows, according to the value of 'k'
    windows = [tokens[i:i+k] for i in range(0,len(tokens)) if len(tokens) >= i+k]
    # print(windows)

    # for every window, get the combinations for the pair of words and count its occurrence
    for window in windows:
        for word1, word2 in combinations(window,2):
            feats["token_pair="+word1+"__"+word2] += 1



neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words. The matching should ignore case.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    """
    ###TODO

    # if there aren't any pos_words or neg_words, the count would be 0
    feats["pos_words"] = 0
    feats["neg_words"] = 0

    # count the total number of pos_words and neg_words considering all the tokens
    for token in tokens:
        if token.lower() in pos_words:
            feats["pos_words"] += 1
        elif token.lower() in neg_words:
            feats["neg_words"] += 1


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    """
    ###TODO

    # Here we update the Feats dictionary, depending on what kind of featurizeation is to be made
    feats = defaultdict(int)
    # call the function in feature_fns list on tokens provided and update the feats
    for func in feature_fns:
        func(tokens, feats)

    return sorted(feats.items())


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index.
    """
    ###TODO

    # Counter to maintain Frequency for documents
    count = Counter()

    # get all the features and featurize them according to the feature function,
    # creating a list of all the featurized tokens
    all_features = []

    for tokens in tokens_list:
        all_features.append(featurize(tokens, feature_fns))
    #print(all_features)


    # Column Count for min_freq, special case of For NegPos Words
    for token, tok_count in chain.from_iterable(all_features):
        if tok_count != 0:
            count[token] +=1

    # here, filtering according to MinFreq requirement
    filtered_minFreq = [token for token, count_token in count.items() if count_token >= min_freq]

    # Creating Vocabulary
    if vocab is None:
        vocab = defaultdict(int)
        for idx, token in enumerate(sorted(filtered_minFreq)):
            vocab[token] = idx
    # print("VOCAB", sorted(vocab.items()))

    # Now, for Generating a CSR Matrix

    # creating some lists for storing the row, column and actual data needed for csr matrix
    colmn, row, data = [], [], []

    # Loop over all the tokens_list and creating a csr matrix accordingly
    for idx, tok in enumerate(tokens_list):
        # feats = featurize(tok, feature_fns)
        for token, count in all_features[idx]:
            # print(token, count)
            if token in vocab:
                row.append(idx) #row no.
                colmn.append(vocab[token]) #Colmn no.
                data.append(count) #data

    row, colmn, data = np.asarray(row), np.asarray(colmn), np.asarray(data)

    # print("ROW: ", len(row), "COL: ", len(colmn), "DATA: ", len(data), "VOCAB: ", len(vocab))

    X = csr_matrix((data, (row, colmn)),dtype=np.int64)

    return X, vocab

def accuracy_score(truth, predicted):
    """ 
	Compute accuracy of predictions.
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation.
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO

    # accuracies for k-folds using cross validation
    accuracies = []

    # generate the Training and Testing data indexes for the each fold
    kf = KFold(len(labels),n_folds=k)

    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        # calculate the accuracy for every fold, finally to get the mean accuracy
        accuracies.append(accuracy_score(y_test, predicted))

        # avg_accuracy += clf.score(X_test, y_test)
    return np.mean(accuracies)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
    """
    ###TODO

    # things needed
    clf = LogisticRegression()
    final_result = []

    # all the combinations for Feature Functions
    comb_feature_fns = []
    for i in range(len(feature_fns)):
        comb_feature_fns.extend(combinations(feature_fns,i+1))
    # print(comb_feature_fns)

    # for every feature function combination
    for punct_val in punct_vals:    # for every punctuation value
        tokens_list = [tokenize(doc, punct_val) for doc in docs] # tokenize the doc
        for comb in comb_feature_fns:   # run the code for every combination of feature func along with different min_freqs
            for freq in min_freqs:
                X, vocab = vectorize(tokens_list, comb, freq, vocab=None)
                avg_accuracy = cross_validation_accuracy(clf, X, labels, 5)
                comb_result = dict()
                comb_result['features'], comb_result['punct'], comb_result['accuracy'], comb_result['min_freq'] = \
                    comb, punct_val, avg_accuracy, freq
                # print(comb_result.items())
                final_result.append(comb_result)

    return sorted(final_result, key=lambda x: -x['accuracy'])


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    """
    ###TODO
    plot_vals = [result['accuracy'] for result in results]
    plt.plot(plot_vals[::-1])
    plt.xlabel("setting")
    plt.ylabel("accuracy")
    #plt.show()
    plt.savefig("accuracies.png")
    #print(plot_vals)

    #pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting.
	
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO

    temp_dict = defaultdict(int)    # a temporary dict generated to store the formatted output
    counter_comb = Counter()        # counter to calculate the total no. of occurrences of each setting
    accuracy_list = {}              # final list of mean accuracy per setting

    # for every result we get from eval all combinations, do
    for result in results:
        # update the dictionary for feature combinations, min_freq and punctuation with their accuracies respectively
        temp_dict["features="+' '.join(val.__name__ for val in result['features'])] +=result['accuracy']
        temp_dict["min_freq="+str(result['min_freq'])]+=result['accuracy']
        temp_dict["punct="+str(result['punct'])] +=result['accuracy']

        # Update the number of occurrences for each setting
        counter_comb["features="+' '' '.join(val.__name__ for val in result['features'])] +=1
        counter_comb["min_freq="+str(result['min_freq'])] +=1
        counter_comb["punct="+str(result['punct'])] +=1

    #print(counter_comb)
    #print(temp_dict)

    # Creating a final dictionary which has mapping for accuracy to its feature setting
    for accuracy, occurrence in zip(sorted(temp_dict.items()), sorted(counter_comb.items())):
        accuracy_list[accuracy[1]/occurrence[1]] = accuracy[0]

    return sorted(accuracy_list.items(),key=lambda x:-x[0])


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    
    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO

    # using the best setting fitting the data again depending on this new setting
    tokens_list = [tokenize(doc, best_result['punct']) for doc in docs]
    X, vocab = vectorize(tokens_list,best_result['features'],best_result['min_freq'])
    clf = LogisticRegression()
    clf.fit(X, labels)

    return clf, vocab

def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    #print(clf.coef_)

    # List for storing the mapping for vocab and its respective coef value
    top_pos_coef = []
    top_neg_coef = []

    # For every pair of vocab_coef value, do
    for val in zip(sorted(vocab), clf.coef_[0]):
        if val[1] >= 0.:
            top_pos_coef.append(val)
        else:
            top_neg_coef.append((val[0], abs(val[1])))

    # return the required positive or negative coef
    if label == 1:
        return sorted(top_pos_coef, key=lambda x: -x[1])[:n]
    else:
        return sorted(top_neg_coef, key=lambda x: -x[1])[:n]



def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.
    
    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO

    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens_list = [tokenize(doc, best_result['punct']) for doc in test_docs]
    X_test, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'], vocab=vocab)

    return test_docs, test_labels, X_test



def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin.
    
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing.
    """
    ###TODO


    proba = clf.predict_proba(X_test)
    store_prob = defaultdict(int)
    predictions = clf.predict(X_test)

    # print(proba)

    # find the indexes where the documents are misclassified
    for idx in range(X_test.shape[0]):
        if predictions[idx] != test_labels[idx]:
            store_prob[idx] = round(proba[idx,predictions[idx]],6)

    # print(sorted(store_prob.items(), key=lambda  x:-x[1]))

    # Now, find N such documents
    for idx, prb in sorted(store_prob.items(), key=lambda x:-x[1])[:n]:
        print("truth="+str(test_labels[idx])+" predicted="+str(predictions[idx])+" proba="+str(prb))
        print(test_docs[idx],end="\n\n")


def main():
    
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Read data.
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    #Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    #Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()