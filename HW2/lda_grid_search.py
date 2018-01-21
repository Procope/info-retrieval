import copy
import itertools
import random
import time
import pyndri
import collections
import io
import logging
import sys
import operator
import pickle
import gensim
import numpy as np

from collections import defaultdict
from math import log, exp
from pprint import pprint

from gensim import corpora, similarities
from gensim.models.ldamodel import LdaModel


from scipy.stats import entropy as kl_divergence


index = pyndri.Index('index/')
token2id, id2token, _ = index.get_dictionary()


def parse_topics(file_or_files, max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    if not isinstance(file_or_files, list) and \
            not isinstance(file_or_files, tuple):
        if hasattr(file_or_files, '__iter__'):
            file_or_files = list(file_or_files)
        else:
            file_or_files = [file_or_files]

    for f in file_or_files:
        assert isinstance(f, io.IOBase)

        for line in f:
            assert(isinstance(line, str))

            line = line.strip()

            if not line:
                continue

            topic_id, terms = line.split(delimiter, 1)

            if topic_id in topics and (topics[topic_id] != terms):
                    logging.error('Duplicate topic "%s" (%s vs. %s).',
                                  topic_id,
                                  topics[topic_id],
                                  terms)

            topics[topic_id] = terms

            if max_topics > 0 and len(topics) >= max_topics:
                break

    return topics


with open('./ap_88_89/topics_title', 'r') as f_topics:
    queries = parse_topics([f_topics])

num_documents = index.maximum_document() - index.document_base()
dictionary = pyndri.extract_dictionary(index)

tokenized_queries = {
    query_id: [dictionary.translate_token(token)
               for token in index.tokenize(query_string)
               if dictionary.has_token(token)]
    for query_id, query_string in queries.items()}

query_term_ids = set(
    query_term_id
    for query_term_ids in tokenized_queries.values()
    for query_term_id in query_term_ids)

print('Gathering statistics about', len(query_term_ids), 'terms.')

# inverted index creation.

document_lengths = {}
unique_terms_per_document = {}

inverted_index = collections.defaultdict(dict)
collection_frequencies = collections.defaultdict(int)

total_terms = 0

start_time = time.time()

for int_doc_id in range(index.document_base(), index.maximum_document()):
    ext_doc_id, doc_token_ids = index.document(int_doc_id)

    document_bow = collections.Counter(
        token_id for token_id in doc_token_ids
        if token_id > 0)
    document_length = sum(document_bow.values())

    document_lengths[int_doc_id] = document_length
    total_terms += document_length

    unique_terms_per_document[int_doc_id] = len(document_bow)

    for query_term_id in query_term_ids:
        assert query_term_id is not None

        document_term_frequency = document_bow.get(query_term_id, 0)

        if document_term_frequency == 0:
            continue

        collection_frequencies[query_term_id] += document_term_frequency
        inverted_index[query_term_id][int_doc_id] = document_term_frequency

avg_doc_length = total_terms / num_documents

print('Inverted index creation took', time.time() - start_time, 'seconds.')


def IDF(query_term_id):
    """
    Inverse Document Frequency of a query term.

    :param query_token_id: the query term id
    """
    df = len(list(inverted_index[query_term_id]))

    return log(num_documents / df)


def tfidf(int_document_id, query_term_id, document_term_freq):
    """
    TF-IDF scoring function for a document and a query term.

    :param int_document_id: the document id
    :param query_token_id: the query term id (assuming you have split the query to tokens)
    :param document_term_freq: the document term frequency of the query term
    """
    return document_term_freq * IDF(query_term_id)


# Define, for convenience, a dictionary mapping external
# document ids to internal document document ids.

ext_to_int = {}
for int_doc_id in range(index.document_base(), index.maximum_document()):
    ext_doc_id, _ = index.document(int_doc_id)
    ext_to_int[ext_doc_id] = int_doc_id

# Read TREC ground truth file containing the VALIDATION SET.
# Create dictionary validation_ids that maps queries to internal and external docID pairs.
# validation_ids: queryID --> (internal_doc_id, external_doc_id)

with open("ap_88_89/qrel_validation", "r") as f:
    validation_ids = {}

    for line in f:
        line = line.strip().split()
        if len(line) != 4:
            continue

        q_id = line[0]
        ext_id = line[2]

        try:
            int_id = ext_to_int[ext_id]

            try:
                validation_ids[q_id].append((int_id, ext_id))
            except KeyError:
                validation_ids[q_id] = [(int_id, ext_id)]

        except KeyError:
            pass  # validation document not in index

texts = []
threshold = 0

for int_doc_id in range(index.document_base(), index.maximum_document()):
    _, token_ids = index.document(int_doc_id)

    text = [
        id2token[token_id]
        for token_id in token_ids
        if collection_frequencies[token_id] > threshold
    ]

    texts.append(text)

dicto = corpora.Dictionary(texts)
corpus = [dicto.doc2bow(text) for text in texts]

lda_models = {}
lda_similarity_indices = {}

start_time = time.time()

for num_topics in np.arange(50, 251, 100):
    lda_models[num_topics] = {}
    lda_similarity_indices[num_topics] = {}

    for chunksize in np.arange(1000, 3001, 1000):
        lda_models[num_topics][chunksize] = {}
        lda_similarity_indices[num_topics][chunksize] = {}

        for alpha in ['symmetric', 'asymmetric', 'auto']:
            lda_models[num_topics][chunksize][alpha] = {}
            lda_similarity_indices[num_topics][chunksize][alpha] = {}

            for eta in [None, 'auto']:
                print('Number of topics: {}. Chunksize: {}. Alpha: {}. Eta: {}'
                      .format(num_topics, chunksize, alpha, eta))

                lda = LdaModel(corpus,
                               num_topics=num_topics,
                               chunksize=chunksize,
                               alpha=alpha,
                               eta=eta,
                               decay=0.5,
                               offset=1.0,
                               eval_every=10,
                               iterations=50,
                               gamma_threshold=0.001,
                               minimum_probability=0.0,
                               minimum_phi_value=0.01,
                               per_word_topics=False)

                lda_models[num_topics][chunksize][alpha][eta] = lda
                lda_similarity_indices[num_topics][chunksize][alpha][eta] = similarities.MatrixSimilarity(
                                                                                        lda[corpus],
                                                                                        num_features=num_topics
                                                                                      )
run_time = int((time.time() - start_time) / 60)
print('Grid search took {} minutes.'.format(run_time))

with open('lda_models.pickle', 'wb') as f:
    pickle.dump(lda_models, f)

with open('lda_similarity_indices.pickle', 'wb') as f:
    pickle.dump(lda_similarity_indices, f)
