import pathlib
import re

import pandas as pd

import tensorflow as tf
import tensorflow_text as text

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")


def preclean(filepath, source, target):
    """

    Reads and preprocesses a CSV file of the bilingual corpus.

    Parameters:
        filepath (str): The location of the bilingual corpus CSV.
        source (str): The language to be translated from.
        target (str): The language to be translated into.

    Returns:
        all_data[source] (list): The list of examples in the source language.
        all_data[target] (list): The list of examples in the target language.
    """

    all_data = pd.read_csv(filepath, encoding="utf-8")

    all_data[source] = all_data[source].str.lower()
    all_data[target] = all_data[target].str.lower()

    all_data = all_data.astype(str)
    
    # Randomize the data
    all_data = all_data.sample(frac=1).reset_index(drop=True)

    # Add a space between to the left of the punctuation
    all_data = all_data.replace('\[', ' [', regex=True)
    all_data = all_data.replace('\]', ' ]', regex=True)
    all_data = all_data.replace('”', ' ”', regex=True)
    all_data = all_data.replace('“', ' “', regex=True)
    all_data = all_data.replace('"', ' "', regex=True)
    all_data = all_data.replace('\(', ' (', regex=True)
    all_data = all_data.replace('\)', ' )', regex=True)
    all_data = all_data.replace('\.', ' .', regex=True)
    all_data = all_data.replace('\?', ' ?', regex=True)
    all_data = all_data.replace('\!', ' !', regex=True)
    all_data = all_data.replace('\,', ' ,', regex=True)
    all_data = all_data.replace('\;', ' ;', regex=True)
    all_data = all_data.replace('\:', ' :', regex=True)

    return all_data[source], all_data[target]


def convert(source_lang, target_lang):
    """Takes the lists of source and target language examples, returns a tensor form of them."""

    return tf.data.Dataset.from_tensor_slices(source_lang), tf.data.Dataset.from_tensor_slices(target_lang)


def generate_vocab(source_lang, target_lang, bert_vocab_args):
    """
    
    Generaters a wordpiece vocabulary of both the source and target language.

    Parameters:
        source_lang (tf.string): The tensor of the source language examples.
        target_lang (tf.string): The tensor of the target language examples.
        bert_vocab_args (dict): A dictionary of the parameters for the tokenizer.

    Return:
        source_vocab (list): A list of strings of wordpieces of the source language.
        target_vocab (list): A list of strings of wordpieces of the target language.
    """

    source_vocab = bert_vocab.bert_vocab_from_dataset(
        source_lang.batch(1000).prefetch(2),
        **bert_vocab_args
    )

    target_vocab = bert_vocab.bert_vocab_from_dataset(
        target_lang.batch(1000).prefetch(2),
        **bert_vocab_args
    )

    return source_vocab, target_vocab


def write_vocab_file(filepath, vocab):
    """Writes the vocabularies into a txt file."""

    with open(filepath, 'w', encoding="utf-8") as file:
        for token in vocab:
            print(token, file=file)


# This function is from the Tensorflow tutorial
def add_start_end(ragged):
    """Concatenates [START] and [END] tokens to examples, returns the concatenated examples."""

    count = ragged.bounding_shape()[0]
    starts = tf.fill([count,1], START)
    ends = tf.fill([count,1], END)
    return tf.concat([starts, ragged, ends], axis=1)


# This function is from the Tensorflow tutorial
def clean(reserved_tokens, token_text):
    """
    
    Drops the reserved tokens and joins the detokenized words.

    Parameters:
        reserved_tokens (list): This is the list of reserved tokens defined at the top.
        token_text (tf.string): The detokenized text.

    Returns:
        result (tf.string): A list of rejoined words in a sentence.
    """

    # Drop the reserved tokens, except for "[UNK]"
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_text, bad_token_re)
    result = tf.ragged.boolean_mask(token_text, ~bad_cells)

    # Join them into strings
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


# These are a class and methods from the Tensorflow tutorial
class CustomTokenizer(tf.Module):
    """
    
    This is a class for the custom tokenizer to tokenize, detokenize, and export the tokenizer.

    Attributes:
        reserved_tokens (list): This is the list of reserved tokens defined at the top.
        vocab_path (string): The location of the bilingual corpus CSV.
    """

    def __init__(self, reserved_tokens, vocab_path):
        """
        
        The constructor for the CustomTokenizer class.

        Parameters:
            reserved_tokens (list): This is the list of reserved tokens defined at the top.
            vocab_path (string): The location of the bilingual corpus CSV.
        """

        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:   

        # Include a tokenize signature for a batch of strings
        self.tokenize.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        """Tokenizes a sentence, returns a tokenized sentence."""
        tokenized = self.tokenizer.tokenize(strings)
        tokenized = tokenized.merge_dims(-2,-1)
        tokenized = add_start_end(tokenized)
        return tokenized

    @tf.function
    def detokenize(self, tokenized):
        """Detokenizes a sentence, returns a detokenized sentence."""
        detokenized = self.tokenizer.detokenize(tokenized)
        return clean(self._reserved_tokens, detokenized)

    @tf.function
    def lookup(self, token_ids):
        """Looks up tokens in vocabulary list, returns the found tokens."""
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        """Gets the size of the vocabulary list, returns the size."""
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        """Gets the location of the vocabulary list, returns the location."""
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        """Gets the reserved tokens, returns the reserved tokens."""
        return tf.constant(self._reserved_tokens)