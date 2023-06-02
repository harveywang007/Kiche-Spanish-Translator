import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# This function is from the Tensorflow tutorial
def split_test_val(source, target, buffer, batch):
    """Splits the data into training and validation sets, returns the training and validation sets."""

    # 60% will be used for training
    training = (
        tf.data.Dataset
        .from_tensor_slices((source[:int(buffer * 0.9)], target[:int(buffer * 0.9)]))
        .shuffle(buffer)
        .batch(batch))

    # 39% will be used for validation
    validation = (
        tf.data.Dataset
        .from_tensor_slices((source[int(buffer * 0.9):int(buffer * 0.99)], target[int(buffer * 0.9):int(buffer * 0.99)]))
        .shuffle(buffer)
        .batch(batch))

    return training, validation


# These are two functions from the Tensorflow tutorial
def make_batches(ds, src_tok, trg_tok, buffer):
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

    return (
            ds
            .shuffle(buffer)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))


def positional_encoding(length, depth):
        """

        Calculates the position vectors for the sentences.

        Parameters:
            length (int): The maximum length of a sentence.
            depth (int): The dimension of the dense layer of the model.

        Returns:
            A tensor with positions encoded in the sentences.
        """

        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1) 

        return tf.cast(pos_encoding, dtype=tf.float32)


# These are a class and methods from the Tensorflow tutorial
class PositionalEmbedding(tf.keras.layers.Layer):
    """

    This is a class for the positional embedding to encode word order. The attention layers in a transformer do not see word order.
    A set of sines and cosines of different frequencies are used to encode the position of each word.
    This layer will look up a token's embedding vector and add a position vector created by the positional_encoding function.

    Attributes:
        d_model (int): The dimension of the dense layer of the model.
        embedding (object): An instance of the embedding class.
        pos_encoding (func): The positional encoding function defined below.
    """

    def __init__(self, vocab_size, d_model, length):
        """

        The constructor for the PositionalEmbedding class.

        Parameters:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the dense layer of the model.
            length (int): The maximum length of a sentence.
        """

        super().__init__()
        self.d_model = d_model

        # Instantiates the Embedding class
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=length, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        """Applies a mask to ignore certain parts of a sentence, returns a masked embedding layer."""

        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        """Applies positional encoding on a tensor, returns the encoded tensor."""

        length = tf.shape(x)[1]
        x = self.embedding(x)

        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


# These are a class and methods from the Tensorflow tutorial
class BaseAttention(tf.keras.layers.Layer):
    """

    This is a class for the base attention layer. The transformer uses multiple attention layers, so this class defines a base class.
    Attention layers all use a multi-head attention layer, a normalization layer, and a layer that adds inputs.
    Other layers will be implemented as subclasses of the base layer.

    Attributes:
        mha (object): An instance of the MultiHeadAttention class.
        layernorm (object): An instance of the LayerNormalization class.
        add (object): An instance of the Add class.
    """

    def __init__(self, **kwargs):
        """

        The constructor for the PositionalEmbedding class.

        Parameters:
            **kwargs: keyword arguments to pass.
        """

        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


# These are a class and methods from the Tensorflow tutorial
class CrossAttention(BaseAttention):
    """

    This is the class for the cross attention layer; it is a subclass of BaseAttention.

    """

    def call(self, x, context):
        """This function implements cross-attention on a sentence, returns the transformed sentence."""

        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


# These are a class and methods from the Tensorflow tutorial
class GlobalSelfAttention(BaseAttention):
    """

    This is the class for the global self-attention layer; it is a subclass of BaseAttention.

    """

    def call(self, x):
        """This function implements global self-attention on a sentence, returns the transformed sentence."""

        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# These are a class and methods from the Tensorflow tutorial
class CausalSelfAttention(BaseAttention):
    """

    This is the class for the global self-attention layer; it is a subclass of BaseAttention.

    """

    def call(self, x):
        """This function implements global self-attention on a sentence, returns the transformed sentence."""

        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


# These are a class and methods from the Tensorflow tutorial
class FeedForward(tf.keras.layers.Layer):
    """

    This is the class for the feed-forward network.

    Attributes:
        seq (object): An instance of the Sequential class.
        add (object): An instance of the Add class.
        layer_norm (object): An instance of the LayerNormalization class.
    """

    def __init__(self, d_model, dff, dropout_rate):
        """

        The constructor for the FeedForward class.

        Parameters:
            d_model (int): The dimension of the outer dense layer.
            dff (int): The dimension of the inner dense layer.
            dropout_rate (float): The drop out rate.
        """

        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        """Applies the feed-foward transformation to the sentence."""

        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x


# These are a class and methods from the Tensorflow tutorial
class EncoderLayer(tf.keras.layers.Layer):
    """

    This is the class for an encoder layer.

    Attributes:
        self_attention (object): An instance of the GlobalSelfAttention class.
        ffn (object): An instance of the FeedForward class.
    """

    def __init__(self, *, d_model, num_heads, dff, gsa_dropout, ff_dropout):
        """

        The constructor for the EncoderLayer class.

        Parameters:
            d_model (int): The dimension of the outer dense layer of the feed-forward layer.
            num_heads (int): The number of attention heads.
            dff (int): The dimension of the inner dense layer of the feed-forward layer.
            gsa_dropout (float): The dropout rate for the global self-attention layer.
            ff_dropout (float): The dropout rate for the feed-forward layer.
        """

        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=gsa_dropout)

        self.ffn = FeedForward(d_model, dff, ff_dropout)

    def call(self, x):
        """Performs the transformations of an encoder layer, returns the transformed sentence."""

        x = self.self_attention(x)
        x = self.ffn(x)
        return x


# These are a class and methods from the Tensorflow tutorial
class Encoder(tf.keras.layers.Layer):
    """

    This is the class for the encoder of the transformer.

    Attributes:
        d_model (int): The dimension of the outer dense layer of the feed-forward layer. 
        num_layers (int): The number of encoder layers.
        pos_embedding (object): An instance of the PositionalEmbedding class.
        enc_layers (list): A list of EncoderLayer objects.
        dropout (object): An instance of the Dropout class.
    """

    def __init__(self, *, num_layers, d_model, length, num_heads,
                 dff, vocab_size, gsa_dropout, ff_dropout, layer_dropout):
        """

        The constructor for the Encoder class.

        Parameters:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimension of the outer dense layer of the feed-forward layer.
            length (int): The maximum length of sentences for positional encoding.
            num_heads (int): The number of attention heads.
            dff (int): The dimension of the inner dense layer of the feed-forward layer.
            vocab_size (int): The size of the vocabulary.
            gsa_dropout (float): The dropout rate for the global self-attention layer.
            ff_dropout (float): The dropout rate for the feed-forward layer.
            layer_dropout (float): The dropout rate for the Dropout class instance.
        """

        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model, length=length)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         gsa_dropout=gsa_dropout,
                         ff_dropout=ff_dropout)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(layer_dropout)

    def call(self, x):
        """Performs the transformations of an encoder, returns the transformed sentence."""

        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)

        x = self.dropout(x) # Add dropout

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


# These are a class and methods from the Tensorflow tutorial
class DecoderLayer(tf.keras.layers.Layer):
    """

    This is the class for a decoder layer.

    Attributes:
        causal_self_attention (object): An instance of the CausalSelfAttention class.
        cross_attention (object): An instance of the CrossAttention class.
        ffn (object): An instance of the FeedForward class.
    """

    def __init__(self, *, d_model, num_heads, dff,
                 csa_dropout, ca_dropout, ff_dropout):
        """
        The constructor for the DecoderLayer class.

        Parameters:
            d_model (int): The dimension of the outer dense layer of the feed-forward layer.
            num_heads (int): The number of attention heads.
            dff (int): The dimension of the inner dense layer of the feed-forward layer.
            csa_dropout (float): The dropout rate for the causal self-attention layer.
            ca_dropout (float): The dropout rate for the cross-attention layer.
            ff_dropout (float): The dropout rate for the feed-forward layer.
        """

        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=csa_dropout)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=ca_dropout)

        self.ffn = FeedForward(d_model, dff, ff_dropout)

    def call(self, x, context):
        """Performs the transformations of a decoder layer, returns the transformed sentence."""

        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


# These are a class and methods from the Tensorflow tutorial
class Decoder(tf.keras.layers.Layer):
    """

    This is the class for the decoder of the transformer.

    Attributes:
        d_model (int): The dimension of the outer dense layer of the feed-forward layer. 
        num_layers (int): The number of decoder layers.
        pos_embedding (object): An instance of the PositionalEmbedding class.
        dropout (object): An instance of the Dropout class.
        dec_layers (list): A list of DecoderLayer objects.
        last_attn_scores (tf.float): A tensor of attention scores.
    """

    def __init__(self, *, num_layers, d_model, length, num_heads,
                 dff, vocab_size, layer_dropout, csa_dropout, ca_dropout, ff_dropout):
        """

        The constructor for the Decoder class.

        Parameters:
            num_layers (int): The number of encoder layers.
            d_model (int): The dimension of the outer dense layer of the feed-forward layer.
            length (int): The maximum length of sentences for positional encoding.
            num_heads (int): The number of attention heads.
            dff (int): The dimension of the inner dense layer of the feed-forward layer.
            vocab_size (int): The size of the vocabulary.
            layer_dropout (float): The dropout rate for the Dropout class instance.
            csa_dropout (float): The dropout rate for the causal self-attention layer.
            ca_dropout (float): The dropout rate for the cross-attention layer.
            ff_dropout (float): The dropout rate for the feed-forward layer.
        """

        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model,
                                                 length=length)
        self.dropout = tf.keras.layers.Dropout(layer_dropout)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, csa_dropout=csa_dropout,
                         ca_dropout=ca_dropout, ff_dropout=ff_dropout)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        """Performs the transformations of a decoder, returns the transformed sentence."""

        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x


# These are a class and methods from the Tensorflow tutorial
class Transformer(tf.keras.Model):
    """

    This is the class for the transformer.

    Attributes:
        encoder (object): An instance of the Encoder class.
        decoder (object): An instance of the Decoder class.
        final_layer (object): An instance of the Dense class.
    """

    def __init__(self, *, num_layers, d_model, length, num_heads, dff,
                 input_vocab_size, target_vocab_size, gsa_dropout,
                 enc_ff_dropout, enc_layer_dropout, dec_layer_dropout,
                 csa_dropout, ca_dropout, dec_ff_dropout):
        """

        The constructor for the Transformer class.

        Parameters:
            num_layers (int): The number of encoder and decoder layers.
            d_model (int): The dimension of the outer dense layer of the feed-forward layer.
            length (int): The maximum length of sentences for positional encoding.
            num_heads (int): The number of attention heads.
            dff (int): The dimension of the inner dense layer of the feed-forward layer.
            input_vocab_size (int): The size of the source language vocabulary.
            target_vocab_size (int): The size of the target language vocabulary.
            gsa_dropout (float): The dropout rate for the global self-attention layer.
            enc_ff_dropout (float): The dropout rate for the feed-forward layer in the encoder layers.
            enc_layer_dropout (float): The dropout rate for the Dropout class instance in the decoder.
            dec_layer_dropout (float): The dropout rate for the Dropout class instance in the encoder.
            csa_dropout (float): The dropout rate for the causal self-attention layer.
            ca_dropout (float): The dropout rate for the cross-attention layer.
            dec_ff_dropout (float): The dropout rate for the feed-forward layer in the decoder layers.
        """

        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               length=length, num_heads=num_heads,
                               dff=dff, vocab_size=input_vocab_size,
                               gsa_dropout=gsa_dropout, ff_dropout=enc_ff_dropout,
                               layer_dropout=enc_layer_dropout)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               length=length, num_heads=num_heads,
                               dff=dff, vocab_size=target_vocab_size,
                               layer_dropout=dec_layer_dropout, csa_dropout=csa_dropout,
                               ca_dropout=ca_dropout, ff_dropout=dec_ff_dropout)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        """Performs the transformations of a transformer, returns the probability values of the predicted tokens and attention weights."""

        context, x = inputs

        context = self.encoder(context)

        x = self.decoder(x, context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


# These are a class and methods from the Tensorflow tutorial
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """

    This is the class for the custom scheduler. This class will adjust the learning rate between epochs.

    Attributes:
        d_model (int): The dimension of the outer dense layer of the feed-forward layer.
        warmup_steps (int): The amount of iterations where the learning rate is lower.
    """

    def __init__(self, d_model, warmup_steps):
        """

        The constructor for the CustomSchedule class.

        Parameters:
            d_model (int): The dimension of the outer dense layer of the feed-forward layer.
            warmup_steps (int): The amount of iterations where the learning rate is lower.
        """

        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Calculates the learning rate, returns the learning rate"""

        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# This function is from the Tensorflow tutorial
def masked_loss(label, pred):
    """Calculates the loss with a padding mask, returns loss."""

    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


# This function is from the Tensorflow tutorial
def masked_accuracy(label, pred):
    """Calculates the accuracy with a padding mask, returns accuracy."""

    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


# These are a class and methods from the Tensorflow tutorial
class Translator(tf.Module):
    """

    This is the class for the translator.

    Attributes:
        src_tok (func): The tokenizer of the source language.
        trg_tok (func): The tokenizer of the target language.
        transformer (object): An instance of the Transformer class.
    """

    def __init__(self, src_tok, trg_tok, transformer):
        """

        The constructor for the Translator class.

        Parameters:
            src_tok (func): The tokenizer of the source language.
            trg_tok (func): The tokenizer of the target language.
            transformer (object): An instance of the Transformer class.
        """
        
        self.src_tok = src_tok
        self.trg_tok = trg_tok
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        """

        Makes the class callable so that the trained transformer can translate.

        Parameters:
            src_tok (func): The tokenizer of the source language.
            trg_tok (func): The tokenizer of the target language.
            transformer (object): An instance of the Transformer class.

        Returns:
            text (str): The translated, detokenized text.
            tokens (list): The corresponding tokens of the text.
            attention_weights (tf.float): A tensor of attention weights.
        """

        # Adds the [START] and [END] tokens to the input sentence.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.src_tok.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # Initializes the output with a [START] token
        start_end = self.trg_tok.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the length dimension
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())

        text = self.trg_tok.detokenize(output)[0]

        tokens = self.trg_tok.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop
        # So, recalculate them outside the loop
        self.transformer([encoder_input, output[:,:-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights


# This function is from the Tensorflow tutorial
def print_translation(sentence, tokens, ground_truth):
    """Prints a sentence in the source language, the model's prediction, and the actual sentence."""

    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


# This function is from the Tensorflow tutorial
def plot_attention_head(in_tokens, translated_tokens, attention):
    """Plots a heatmap of one attention head."""

    # The model didn't generate `<START>` in the output
    # Skip it
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(
        labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


# This function is from the Tensorflow tutorial
def plot_attention_weights(sentence, src_tok, trg_tok, translated_tokens, attention_heads):
    """Plots all of the attention heads."""

    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = src_tok.tokenize(in_tokens).to_tensor()
    in_tokens = trg_tok.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()


# These are a class and methods from the Tensorflow tutorial
class ExportTranslator(tf.Module):
    """

    This is the class for exporting the translator. The trained translator can be called and used from this class.

    Attributes:
        translator (object): An instance of the translator class.
    """

    def __init__(self, translator):
        """

        The constructor for the ExportTranslator class.

        Parameters:
            translator (object): An instance of the translator class.
        """

        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        """

        Makes the class callable so that it can translate the sentences using the trained model.

        Parameters:
            sentence (str): A sentence to be translated.

        Returns:
            result (str): A translated sentence.
        """

        (result, tokens, attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

        return result