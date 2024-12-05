import os
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import streamlit as st

# Constants
LSTM_MODEL_SAVE_PATH = 'lstm_chatbot_model'
LSTM_VOCAB_SAVE_PATH = 'lstm_chatbot_vocab.pkl'

RNN_MODEL_SAVE_PATH = 'rnn_chatbot_model'
RNN_VOCAB_SAVE_PATH = 'rnn_chatbot_vocab.pkl'

GRU_MODEL_SAVE_PATH = 'gru_chatbot_model'
GRU_VOCAB_SAVE_PATH = 'gru_chatbot_vocab.pkl'

BILSTM_MODEL_SAVE_PATH = 'bilstm_chatbot_model'
BILSTM_VOCAB_SAVE_PATH = 'bilstm_chatbot_vocab.pkl'

input_seq_len = 15
output_seq_len = input_seq_len + 2
marker_start = '<begin>'
marker_end = '<end>'
marker_pad = '<pad>'
marker_unknown = '<unk>'

data_options = ['Головна', 'LSTM Chatbot', 'GRU Chatbot', 'RNN Chatbot', 'BiLSTM Chatbot']
app_mode = st.sidebar.selectbox('Menu', options=data_options)

# Load Vocabulary Mappings
def load_vocab_mappings(model_type):
    vocab_path = {
        'LSTM Chatbot': LSTM_VOCAB_SAVE_PATH,
        'GRU Chatbot': GRU_VOCAB_SAVE_PATH,
        'RNN Chatbot': RNN_VOCAB_SAVE_PATH,
        'BiLSTM Chatbot': BILSTM_VOCAB_SAVE_PATH,
    }.get(model_type)
    
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        return (
            vocab_data['input_w2i'], 
            vocab_data['input_i2w'], 
            vocab_data['target_w2i'], 
            vocab_data['target_i2w']
        )
    return None

# Load Model
@st.cache_resource
def load_model(model_type):
    model_path = {
        'LSTM Chatbot': LSTM_MODEL_SAVE_PATH,
        'GRU Chatbot': GRU_MODEL_SAVE_PATH,
        'RNN Chatbot': RNN_MODEL_SAVE_PATH,
        'BiLSTM Chatbot': BILSTM_MODEL_SAVE_PATH,
    }.get(model_type)
    
    if model_path and os.path.exists(model_path):
        model = tf.saved_model.load(model_path)
        print(f"{model_type} model loaded successfully.")
        return model
    return None


# Preprocess Input
def preprocess_input(input_msg, input_w2i):
    msg_lower = [w.lower() for w in nltk.word_tokenize(input_msg)]
    msg = [w for w in msg_lower if any(char.isalnum() for char in w)]
    
    if len(msg) > input_seq_len:
        msg = msg[:input_seq_len]
    
    human_msg_encoded = [input_w2i.get(word, 0) for word in msg]
    human_msg_encoded = human_msg_encoded + [input_w2i[marker_pad]] * (input_seq_len - len(human_msg_encoded))
    
    return np.array([human_msg_encoded])

# Generate Reply
def generate_reply(model, input_msg, input_w2i, target_w2i, target_i2w):
    # Preprocess the input message
    msg = [w.lower() for w in nltk.word_tokenize(input_msg) if any(c.isalnum() for c in w)]
    msg_encoded = [input_w2i.get(w, 0) for w in msg]
    encoder_input = pad_sequences([msg_encoded], maxlen=input_seq_len, padding='post')

    # Initialize the decoder input
    decoder_input = np.zeros((1, output_seq_len - 1), dtype=int)
    decoder_input[0, 0] = target_w2i[marker_start]

    for i in range(1, output_seq_len - 1):
        inputs = {
            "encoder_input": tf.convert_to_tensor(encoder_input, dtype=tf.int32),
            "decoder_input": tf.convert_to_tensor(decoder_input, dtype=tf.int32)
        }
        outputs = model.signatures["serving_default"](**inputs)
        preds = outputs["output"].numpy()  # Adjust based on your model's output key
        pred_idx = np.argmax(preds[0, i - 1])
        decoder_input[0, i] = pred_idx
        if pred_idx == target_w2i[marker_end]:
            break

    # Decode output to words
    reply = ' '.join(target_i2w[idx] for idx in decoder_input[0] if idx not in [target_w2i[marker_start], target_w2i[marker_end], target_w2i[marker_pad]])
    return reply

# Main Function
def main():
    # Clear message history on page switch
    if "current_page" not in st.session_state:
        st.session_state.current_page = app_mode  # Initialize the current page tracker

    if st.session_state.current_page != app_mode:
        st.session_state.current_page = app_mode  # Update the current page tracker
        st.session_state.messages = []  # Clear message history on page switch

    # Dynamic Title
    if app_mode == "Головна":
        st.title("Welcome to the Mental Health Chat Bot")
        st.write("Select a chatbot model from the sidebar to begin.")
        return
    else:
        st.title(f"{app_mode} - Mental Health Chat Bot")
    
    # Load model and vocab for the selected app_mode
    model = load_model(app_mode)
    vocab_mappings = load_vocab_mappings(app_mode)
    if not model or not vocab_mappings:
        st.error(f"{app_mode} model or vocabulary mappings not found. Please train the model first.")
        return
    
    input_w2i, input_i2w, target_w2i, target_i2w = vocab_mappings
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input and response handling
    if user_input := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate bot reply
        bot_reply = generate_reply(model, user_input, input_w2i, target_w2i, target_i2w)
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        # Add bot reply to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

if __name__ == "__main__":
    main()
