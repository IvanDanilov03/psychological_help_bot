import os
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import streamlit as st
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
import matplotlib.pyplot as plt

# Configuration and Constants
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
DATA_DIR_PATH = 'data'
MAX_VOCAB_SIZE = 30000

MODEL_SAVE_PATH = 'bilstm_chatbot_model'
VOCAB_SAVE_PATH = 'bilstm_chatbot_vocab.pkl'
EVALUATION_SAVE_PATH = 'bilstm_evaluation_results.json'
TRAINING_LOSS_PLOT_SAVE_PATH = 'bilstm_training_loss.png'

marker_start = '<begin>'
marker_end = '<end>'
marker_unknown = '<unk>'
marker_pad = '<pad>'

input_seq_len = 15
output_seq_len = input_seq_len + 2

# Data Preprocessing
def load_and_preprocess_data():
    target_counter = Counter()
    input_counter = Counter()

    input_texts = []
    target_texts = []

    for file in os.listdir(DATA_DIR_PATH):
        filepath = os.path.join(DATA_DIR_PATH, file)
        if os.path.isfile(filepath):
            print(f'Processing file: {file}')
            lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
            prev_words = []

            for line in lines:
                if line.startswith('- - '):
                    prev_words = []

                if line.startswith('- - ') or line.startswith('  - '):
                    line = line.replace('- - ', '').replace('  - ', '')
                    next_words = [w.lower() for w in nltk.word_tokenize(line)]
                    next_words = [w for w in next_words if any(c.isalnum() for c in w)]

                    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
                        next_words = next_words[:MAX_TARGET_SEQ_LENGTH]

                    if prev_words:
                        input_texts.append(prev_words)
                        for w in prev_words:
                            input_counter[w] += 1

                        target_words = next_words[:]
                        for w in target_words:
                            target_counter[w] += 1
                        target_texts.append(target_words)

                    prev_words = next_words

    # Create word-to-index mappings
    input_w2i = {marker_unknown: 0, marker_pad: 1}
    target_w2i = {marker_unknown: 0, marker_pad: 1, marker_start: 2, marker_end: 3}

    for idx, (word, _) in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
        input_w2i[word] = idx + 2

    for idx, (word, _) in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
        target_w2i[word] = idx + 4

    input_i2w = {idx: word for word, idx in input_w2i.items()}
    target_i2w = {idx: word for word, idx in target_w2i.items()}

    # Convert to numerical sequences
    x = [[input_w2i.get(word, 0) for word in sentence] for sentence in input_texts]
    y = [[target_w2i.get(word, 0) for word in sentence] for sentence in target_texts]

    x = pad_sequences(x, maxlen=input_seq_len, padding='post', truncating='post')
    y = [
        pad_sequences(
            [[target_w2i[marker_start]] + sentence + [target_w2i[marker_end]]],
            maxlen=output_seq_len,
            padding='post',
            truncating='post'
        )[0] for sentence in y
    ]

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.05)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test, input_w2i, input_i2w, target_w2i, target_i2w

# Model Definition with Bidirectional LSTM
class Seq2SeqWithAttentionBidirectional(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, embedding_dim=100, hidden_dim=512):
        super(Seq2SeqWithAttentionBidirectional, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)

        # Encoder: Bidirectional LSTM
        self.encoder_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        )

        # Project encoder outputs and states to decoder's hidden dimension
        self.encoder_projection = tf.keras.layers.Dense(hidden_dim)
        self.state_projection_h = tf.keras.layers.Dense(hidden_dim, activation="tanh")
        self.state_projection_c = tf.keras.layers.Dense(hidden_dim, activation="tanh")

        # Attention mechanism
        self.attention = tf.keras.layers.AdditiveAttention()

        # Decoder LSTM
        self.decoder_lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True)

        # Output dense layer
        self.output_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

    def call(self, inputs):
        encoder_input, decoder_input = inputs

        # Encoder processing
        encoder_embedded = self.encoder_embedding(encoder_input)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(encoder_embedded)

        # Project encoder outputs to match decoder hidden dimensions
        encoder_outputs = self.encoder_projection(encoder_outputs)

        # Concatenate the forward and backward states
        state_h = tf.concat([forward_h, backward_h], axis=-1)
        state_c = tf.concat([forward_c, backward_c], axis=-1)

        # Map the concatenated states to the decoder's expected size
        decoder_state_h = self.state_projection_h(state_h)
        decoder_state_c = self.state_projection_c(state_c)

        # Decoder processing
        decoder_embedded = self.decoder_embedding(decoder_input)
        decoder_outputs, _, _ = self.decoder_lstm(
            decoder_embedded, initial_state=[decoder_state_h, decoder_state_c]
        )

        # Attention mechanism
        attention_context = self.attention([decoder_outputs, encoder_outputs])
        combined_context = tf.concat([decoder_outputs, attention_context], axis=-1)

        # Output predictions
        logits = self.output_layer(combined_context)

        return {"output": logits}


# Model Evaluation
class ModelEvaluator:
    def __init__(self, model_fn, input_w2i, target_i2w, target_w2i, max_seq_len):
        self.model_fn = model_fn
        self.input_w2i = input_w2i
        self.target_i2w = target_i2w
        self.target_w2i = target_w2i
        self.max_seq_len = max_seq_len
        self.smooth_fn = SmoothingFunction().method1

    def calculate_bleu(self, input_sequences, target_sequences):
        total_bleu = 0
        for input_seq, target_seq in zip(input_sequences, target_sequences):
            predicted_seq = self.generate_reply(input_seq)
            reference = [self.target_i2w[idx] for idx in target_seq if idx not in [0, 1]]
            total_bleu += sentence_bleu([reference], predicted_seq, smoothing_function=self.smooth_fn)
        return total_bleu / len(input_sequences)

    def calculate_rouge(self, input_sequences, target_sequences):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []
        for input_seq, target_seq in zip(input_sequences, target_sequences):
            predicted_seq = self.generate_reply(input_seq)
            reference = ' '.join([self.target_i2w[idx] for idx in target_seq if idx not in [0, 1]])
            prediction = ' '.join(predicted_seq)
            scores.append(scorer.score(reference, prediction))
        avg_scores = {metric: np.mean([score[metric].fmeasure for score in scores]) for metric in ['rouge1', 'rouge2', 'rougeL']}
        return avg_scores

    def calculate_perplexity(self, input_sequences, target_sequences):
        log_probs = []
        for input_seq, target_seq in zip(input_sequences, target_sequences):
            # Get predicted probabilities
            predicted_probs = self.model_fn(
                encoder_input=tf.convert_to_tensor([input_seq], dtype=tf.int32),
                decoder_input=tf.convert_to_tensor([target_seq[:-1]], dtype=tf.int32)
            )["output"].numpy()  # Shape: (1, sequence_length, vocab_size)

            # Align target indices with predicted_probs
            target_seq_aligned = target_seq[1:1 + predicted_probs.shape[1]]  # Align lengths
            target_indices = np.expand_dims(target_seq_aligned, axis=-1)  # Shape: (sequence_length, 1)

            # Extract probabilities of the correct target tokens
            target_probs = np.take_along_axis(predicted_probs[0], target_indices, axis=-1).squeeze(-1)  # Shape: (sequence_length,)
            
            # Avoid log(0) by filtering valid probabilities
            valid_probs = target_probs[target_probs > 0]
            log_probs.extend(np.log(valid_probs))  # Log probabilities for perplexity

        return np.exp(-np.mean(log_probs)) if log_probs else float('inf')  # Handle empty log_probs

    def generate_reply(self, input_seq):
        decoder_input = np.zeros((1, self.max_seq_len - 1), dtype=int)
        decoder_input[0, 0] = self.target_w2i['<begin>']

        for i in range(1, self.max_seq_len - 1):
            inputs = {
                "encoder_input": tf.convert_to_tensor([input_seq], dtype=tf.int32),
                "decoder_input": tf.convert_to_tensor(decoder_input, dtype=tf.int32),
            }
            outputs = self.model_fn(**inputs)
            pred_idx = np.argmax(outputs["output"].numpy()[0, i - 1])
            decoder_input[0, i] = pred_idx
            if pred_idx == self.target_w2i['<end>']:
                break

        return [self.target_i2w[idx] for idx in decoder_input[0] if idx not in [0, 1, 2, 3]]

    def evaluate(self, X_test, Y_test, save_path=EVALUATION_SAVE_PATH):
        print("Evaluating model...")

        # Check if evaluation results already exist
        if os.path.exists(save_path):
            print(f"Loading saved evaluation results from {save_path}")
            with open(save_path, 'r') as f:
                results = json.load(f)
            return results

        # Run evaluation
        bleu = self.calculate_bleu(X_test, Y_test)
        print(f"BLEU Score: {bleu:.4f}")

        rouge = self.calculate_rouge(X_test, Y_test)
        print(f"ROUGE Scores: {rouge}")

        perplexity = self.calculate_perplexity(X_test, Y_test)
        print(f"Perplexity: {perplexity:.4f}")

        # Combine all results
        results = {
            "bleu": float(bleu),  # Convert to native Python float
            "rouge": {k: float(v) for k, v in rouge.items()},  # Convert Rouge scores
            "perplexity": float(perplexity),
        }

        # Save results to a file
        print(f"Saving evaluation results to {save_path}")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results
    

# Plot
def plot_training_loss(history):
    """Plot training and validation loss and save the plot."""
    plt.figure(figsize=(8, 6))
    
    # Check if history contains valid keys
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        # Save the plot as an image
        plt.savefig(TRAINING_LOSS_PLOT_SAVE_PATH, dpi=300)
        print(f"Training loss plot saved as {TRAINING_LOSS_PLOT_SAVE_PATH}")
    else:
        print("History object does not contain 'loss' or 'val_loss'. Unable to plot training loss.")
    
    # Explicitly close the plot to flush it
    plt.close()


# Training
def train_model_with_bilstm(X_train, Y_train, input_vocab_size, target_vocab_size):
    model = Seq2SeqWithAttentionBidirectional(input_vocab_size, target_vocab_size)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.007),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        [X_train, Y_train[:, :-1]], Y_train[:, 1:],
        batch_size=64,
        epochs=50,
        validation_split=0.2
    )

    # Plot the training loss
    plot_training_loss(history)

    return model

def save_vocab_mappings(input_w2i, input_i2w, target_w2i, target_i2w):
    vocab_data = {
        'input_w2i': input_w2i,
        'input_i2w': input_i2w,
        'target_w2i': target_w2i,
        'target_i2w': target_i2w
    }
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab_data, f)


def load_vocab_mappings():
    if os.path.exists(VOCAB_SAVE_PATH):
        with open(VOCAB_SAVE_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        return (
            vocab_data['input_w2i'], 
            vocab_data['input_i2w'], 
            vocab_data['target_w2i'], 
            vocab_data['target_i2w']
        )
    return None


# Save Model with Signature
def save_model_with_signature(model):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, input_seq_len), dtype=tf.int32, name="encoder_input"),
        tf.TensorSpec(shape=(None, output_seq_len - 1), dtype=tf.int32, name="decoder_input")
    ])
    def call_model(encoder_input, decoder_input):
        return model([encoder_input, decoder_input])
    
    tf.saved_model.save(model, MODEL_SAVE_PATH, signatures={"serving_default": call_model})

# Load Saved Model
def load_saved_model():
    return tf.saved_model.load(MODEL_SAVE_PATH).signatures["serving_default"]

# Response Generation
def generate_reply(model_fn, input_msg, input_w2i, target_w2i, target_i2w):
    msg = [w.lower() for w in nltk.word_tokenize(input_msg) if any(c.isalnum() for c in w)]
    msg_encoded = [input_w2i.get(w, 0) for w in msg]
    msg_padded = pad_sequences([msg_encoded], maxlen=input_seq_len, padding='post')

    decoder_input = np.zeros((1, output_seq_len - 1), dtype=int)
    decoder_input[0, 0] = target_w2i[marker_start]

    for i in range(1, output_seq_len - 1):
        inputs = {
            "encoder_input": tf.convert_to_tensor(msg_padded, dtype=tf.int32),
            "decoder_input": tf.convert_to_tensor(decoder_input, dtype=tf.int32)
        }
        outputs = model_fn(**inputs)
        preds = outputs["output"].numpy()  # Explicitly use the named output
        pred_idx = np.argmax(preds[0, i - 1])
        decoder_input[0, i] = pred_idx
        if pred_idx == target_w2i[marker_end]:
            break

    reply = ' '.join(target_i2w[idx] for idx in decoder_input[0] if idx not in [target_w2i[marker_start], target_w2i[marker_end], target_w2i[marker_pad]])
    return reply

# Main Function for Training and Streamlit
def main():
    print("Chatbot Training and Evaluation")

    # Check if a pre-trained model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading pre-trained model...")
        model_fn = load_saved_model()
        vocab_mappings = load_vocab_mappings()
        if vocab_mappings:
            input_w2i, input_i2w, target_w2i, target_i2w = vocab_mappings
        else:
            print("Vocabulary not found! Exiting...")
            return
        # Load data for evaluation
        _, _, X_test, Y_test, _, _, _, _ = load_and_preprocess_data()
    else:
        # Train the model if no pre-trained model exists
        print("Training new model...")
        X_train, Y_train, X_test, Y_test, input_w2i, input_i2w, target_w2i, target_i2w = load_and_preprocess_data()
        model = train_model_with_bilstm(X_train, Y_train, len(input_w2i), len(target_w2i))
        save_model_with_signature(model)
        save_vocab_mappings(input_w2i, input_i2w, target_w2i, target_i2w)
        model_fn = model  # Use the newly trained model

    # Evaluate the model if test data is available
    if X_test is not None and Y_test is not None:
        print("Evaluating the model...")
        evaluator = ModelEvaluator(
            model_fn=model_fn,
            input_w2i=input_w2i,
            target_i2w=target_i2w,
            target_w2i=target_w2i,
            max_seq_len=output_seq_len
        )
        evaluation_results = evaluator.evaluate(X_test, Y_test)
        print("Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric.capitalize()}: {value:.4f}" if isinstance(value, float) else f"{metric.capitalize()}: {value}")

if __name__ == "__main__":
    main()
