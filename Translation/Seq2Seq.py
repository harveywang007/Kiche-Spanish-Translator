import pandas as pd


import einops

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf


MAX_TOKENS = 128


def preclean(filepath, source, target):
    """

    Reads and preprocesses a CSV file of the bilingual corpus.

    Parameters:
        filepath (str): The location of the bilingual corpus CSV.
        source (str): The language to be translated from.
        target (str): The language to be translated into.

    Returns:
        The list of examples in the source language.
        The list of examples in the target language.
    """

    all_data = pd.read_csv(filepath, encoding="utf-8")

    all_data[source] = all_data[source].str.lower()
    all_data[target] = all_data[target].str.lower()

    all_data = all_data.astype(str)
    all_data = all_data.sample(frac=1).reset_index(drop=True)

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


# These are a class and methods from the Tensorflow tutorial
class ShapeChecker():
    """

    This is the class for checking the shapes of tensors. The low-level APIs used leave much room for confusion with tensor shapes.

    Attributes:
        shapes (dict): A dictionary containing the dimensions of tensors.
    """

    def __init__(self):
        """

        The constructor for the ShapeChecker class.

        Parameters:
            shapes (dict): A dictionary containing the dimensions of tensors.
        """

        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        """

        Makes the class callable.

        Parameters:
            tensor (tf.float): The tensor to be checked.
            names (str): The name of the tensor.

        Returns:
            None.

        Raises:
            ValueError: If the old and new dimensions do not match.
        """

        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:

                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")


# This function is from the Tensorflow tutorial
def split_test_val(source, target, buffer, batch):
    """Splits the data into training and validation sets, returns the training and validation sets."""

    training = (
        tf.data.Dataset
        .from_tensor_slices((source[:int(buffer * 0.9)], target[:int(buffer * 0.9)]))
        .shuffle(buffer)
        .batch(batch))

    validation = (
        tf.data.Dataset
        .from_tensor_slices((source[int(buffer * 0.9):int(buffer * 0.99)], target[int(buffer * 0.9):int(buffer * 0.99)]))
        .shuffle(buffer)
        .batch(batch))

    return training, validation


# These are two functions from the Tensorflow tutorial
def make_batches(ds, src_tok, trg_tok):
    """

    Converts the examples into batches. This function contains a nested function that will prepare the examples for batching.

    Parameters:
        ds (tf.string): The source and target examples to be split into batches.
        src_tok (func): The tokenizer for the source language.
        trg_tok (func): The tokenizer for the target language.
        buffer (int): This number affects the randomness of shuffling.

    Returns:
        The batched dataset.
    """

    def prepare_batch(source, target):
        """

        Tokenizes, trims, and pads texts. The target tokens are duplicated make inputs and labels;
        they are shifted one step so the label is the ID of the next token at each input location.

        Parameters:
            source (tf.string): A tensor containing examples of the language to be translated from.
            target (tf.string): A tensors containing examples of the language to be translated into.

        Returns:
            source (tf.string): A tensor containing tokenized, trimmed, and padded examples of the source language.
            trg_inputs (tf.string): A tensor containing tokenized and padded examples of the target language, whose [END] token is trimmed off.
            trg_labels (tf.string): A tensor containing tokenized and padded examples of the target language, whose [START] token is trimmed off.
        """

        source = src_tok.tokenize(source) # Output is ragged
        source = source[:, :MAX_TOKENS] # Trim to MAX_TOKENS
        source = source.to_tensor() # Convert to 0-padded dense Tensor

        target = trg_tok.tokenize(target)
        target = target[:, :(MAX_TOKENS+1)]
        trg_inputs = target[:, :-1].to_tensor() # Drop the [END] tokens
        trg_labels = target[:, 1:].to_tensor() # Drop the [START] tokens

        return (source, trg_inputs), trg_labels

    return (ds.map(prepare_batch, tf.data.AUTOTUNE))


# These are a class and methods from the Tensorflow tutorial
class Encoder(tf.keras.layers.Layer):
    """

    This is the class for the encoder of the model.

    Attributes:
        tokenizer (func): The tokenizer of the source language.
        vocab_size (int): The size of the vocabulary.
        units (int): The amount of units in the GRU.
        embedding (object): An instance of the Embedding class.
        rnn (object): An instance of the Bidirectional class.
    """

    def __init__(self, tokenizer, vocab_size, units, dropout_rate):
        """

        The constructor for the Encoder class.

        Parameters:
            tokenizer (func): The tokenizer of the source language.
            vocab_size (int): The size of the vocabulary.
            units (int): The amount of units in the GRU.
            dropout_rate (float): The dropout rate of the GRU.
        """

        super(Encoder, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.units = units

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        # The RNN layer processes those vectors sequentially.
        self.rnn = tf.keras.layers.Bidirectional(merge_mode='sum',
                                                 layer=tf.keras.layers.GRU(units,
                                                                           # Return the sequence and state
                                                                           return_sequences=True,
                                                                           dropout=dropout_rate,
                                                                           recurrent_initializer='glorot_uniform'))

    def call(self, x):
        """Performs the transformations of an encoder, returns the transformed sentence."""

        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        # 2. The embedding layer looks up the embedding vector for each token.
        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        # 3. The GRU processes the sequence of embeddings.
        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        # 4. Returns the new sequence of embeddings.
        return x

    def convert_input(self, texts):
        """Converts a list of strings to tensors, returns the converted text."""

        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.tokenizer(texts).to_tensor()
        context = self(context)
        return context


# These are a class and methods from the Tensorflow tutorial
class CrossAttention(tf.keras.layers.Layer):
    """

    This is the class for the attention of the model.

    Attributes:
        mha (object): An instance of the MultiHeadAttention class.
        layernorm (object): An instance of the LayerNormalization class.
        add (object): An instance of the Add class.
    """

    def __init__(self, units, dropout, num_heads, **kwargs):
        """

        The constructor for the CrossAttention class.

        Parameters:
            units (int): The dimension of the dense layer of the model.
            dropout (float): The dropout rate of the GRU.
            num_heads (int): The number of attention heads.
            **kwargs: keyword arguments to pass.
        """

        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, dropout=dropout,
                                                      num_heads=num_heads, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        """Performs the transformations of an attention head, returns the transformed sentence."""

        shape_checker = ShapeChecker()

        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        attn_output, attn_scores = self.mha(
                                            query=x,
                                            value=context,
                                            return_attention_scores=True)

        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


# These are a class and methods from the Tensorflow tutorial
class Decoder(tf.keras.layers.Layer):
    """

    This is the class for the decoder of the model.

    Attributes:
        detokenizer (func): The detokenizer of the target language.
        vocab_size (int): The size of the vocabulary.
        vocab (list): A list of strings in the vocabulary.
        units (int): The amount of units in the GRU.
        word_to_id (object): An instance of StringLookup.
        start_token (int): The start token tokenized.
        end_token (int): The end token tokenized.
        embedding (object): An instance of the Embedding class.
        rnn (object): An instance of the Bidirectional class.
        attention (object): An instance of the CrossAttention class.
        output_layer (object): An instance of the Dense class.
    """

    def __init__(self, detokenizer, vocab_size, vocab,
                 units, gru_dropout, att_dropout, num_heads):
        """

        The constructor for the Decoder class.

        Parameters:
            detokenizer (func): The detokenizer of the target language.
            vocab_size (int): The size of the vocabulary.
            vocab (list): A list of strings in the vocabulary.
            units (int): The amount of units in the GRU.
            gru_dropout (float): The dropout rate for the GRU layer.
            att_dropout (float): The dropout rate for the attention layer.
            num_heads (int): The number of attention heads.
        """

        super(Decoder, self).__init__()
        self.detokenizer = detokenizer
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.units = units

        self.word_to_id = tf.keras.layers.StringLookup(vocabulary=self.vocab,
                                                       mask_token='', oov_token='[UNK]')

        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        # 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform',
                                       dropout=gru_dropout)

        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units, att_dropout, num_heads)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        """Performs the transformations of a decoder, returns the transformed sentence."""

        shape_checker = ShapeChecker()
        shape_checker(x, 'batch t')
        shape_checker(context, 'batch s units')

        # 1. Lookup the embeddings
        x = self.embedding(x)
        shape_checker(x, 'batch t units')

        # 2. Process the target sequence.
        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, 'batch t units')

        # 3. Use the RNN output as the query for the attention over the context.
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, 'batch t units')
        shape_checker(self.last_attention_weights, 'batch t s')

        # Step 4. Generate logit predictions for the next token.
        logits = self.output_layer(x)
        shape_checker(logits, 'batch t target_vocab_size')

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        """Gets the initial state of the RNN, returns the start token, a tensor of falses, and the initial state."""

        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        """Detokenizes a string of tokens, returns the detokenized text."""

        return self.detokenizer(tokens)

    def get_next_token(self, context, next_token, done, state, temperature):
        """Predicts the next token, returns the token, a tensor of booleans, and the state."""

        logits, state = self(context, next_token, state=state, return_state=True)

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :]/temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        # If a sequence produces an `end_token`, set it `done`
        done = done | (next_token == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state


# These are a class and methods from the Tensorflow tutorial
class Translator(tf.keras.Model):
    """

    This is the class for the translator.

    Attributes:
        encoder (object): An instance of the Encoder class.
        decoder (object): An instance of the Decoder class.
    """

    def __init__(self, units, tokenizer, detokenizer,
                 src_size, tar_size, vocab, enc_dropout,
                 dec_gru_drop, dec_att_drop, num_heads):
        """

        The constructor for the Translator class.

        Parameters:
            units (int): The amount of units in the GRU.
            tokenizer (func): The tokenizer of the source language.
            detokenizer (func): The detokenizer of the target language.
            src_size (int): The size of the source language's vocabulary.
            tar_size (int): The size of the target language's vocabulary.
            vocab (list): A list of strings in the vocabulary of the target language.
            enc_dropout (float): The dropout rate for the encoder's GRU.
            dec_gru_dropout (float): The dropout rate for the decoder's GRU layer.
            dec_att_dropout (float): The dropout rate for the decoder's attention layer.
            num_heads (int): The number of attention heads.
        """

        super().__init__()
        encoder = Encoder(tokenizer, src_size, units, enc_dropout)
        decoder = Decoder(detokenizer, tar_size, vocab, units, dec_gru_drop, dec_att_drop, num_heads)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """Runs sentence through the encoder and decoder, returns the probabilities of tokens."""

        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        return logits

    def translate(self, texts, *, max_length=128, temperature=0.0):
        """
        
        Translates the sentence, from string to string.

        Parameters:
            texts (list): A list of strings containing the texts to be translated.

        Returns:
            result (str): The translation.
        """

        # Process the input texts
        context = self.encoder.convert_input(texts)

        # Process the input texts
        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        for _ in range(max_length):
            # Generate the next token
            next_token, done, state = self.decoder.get_next_token(context, next_token,
                                                                  done, state, temperature)

            # Collect the generated tokens
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Stack the lists of tokens and attention weights.
        tokens = tf.concat(tokens, axis=-1)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)

        result = self.decoder.tokens_to_text(tokens)
        return result

    def plot_attention(self, text, **kwargs):
        """Plots a heatmap of one attention head."""

        assert isinstance(text, str)
        output = self.translate([text], **kwargs)
        output = output[0].numpy().decode()

        attention = self.last_attention_weights[0]

        context = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
        context = tf.strings.regex_replace(context, '[.?!,¿]', r' \0 ')
        context = tf.strings.strip(context)
        context = context.numpy().decode().split()

        output = tf.strings.regex_replace(output, '[^ a-z.?!,¿]', '')
        output = tf.strings.regex_replace(output, '[.?!,¿]', r' \0 ')
        output = tf.strings.strip(output)
        output = output.numpy().decode().split()[1:]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        ax.matshow(attention, cmap='viridis', vmin=0.0)

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + output, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax.set_xlabel('Input text')
        ax.set_ylabel('Output text')


def masked_loss(y_true, y_pred):
    """Calculates the loss with a padding mask, returns loss."""

    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    """Calculates the accuracy with a padding mask, returns accuracy."""

    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

# These are a class and methods from the Tensorflow tutorial
class Export(tf.Module):
    """

    This is the class for exporting the translator.

    Attributes:
        model (object): An instance of the translator class.
    """

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        """Translates the a sentence, returns the translation."""

        return self.model.translate(inputs)