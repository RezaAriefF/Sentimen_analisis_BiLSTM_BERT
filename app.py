import re
import os
import bert
import tensorflow_hub as hub
import tensorflow as tf
import streamlit as st
from transformers import AutoTokenizer


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TFHUB_CACHE_DIR"] = "some_dir"

# Loading model
model = tf.keras.models.load_model('modelv6.h5', custom_objects={ 'KerasLayer': hub.KerasLayer})
model.make_predict_function()

#load tokenizer
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer('bert_multi_cased_L-12_H-768_A-12_4', trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# This is a path to an uncased (all lowercase) version of BERT
# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

# def create_tokenizer_from_hub_module():
# #   """Get the vocab file and casing info from the Hub module."""
#   with tf.Graph().as_default():
#     bert_module = hub.Module(BERT_MODEL_HUB)
#     tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
#     with tf.Session() as sess:
#       vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
#                                             tokenization_info["do_lower_case"]])
      
#   return bert.tokenization.FullTokenizer(
#       vocab_file=vocab_file, do_lower_case=do_lower_case)

# tokenizer = create_tokenizer_from_hub_module()

def case_fold(text) :
    case_fold = text.lower()
    return case_fold

def remove_punc (case_fold) :
    remove_punc = re.sub(r'[^\w\s]', '', case_fold)
    return remove_punc

def remove_stop(remove_punc):
    text_tokens = word_tokenize(remove_punc)
    tokens_without_sw = [word for word in text_tokens 
    if not word in stopwords.words(['english', 'portuguese', 'french', 'spanish', 'russian', 'italian', 'german'])]
    remove_stop = (" ").join(tokens_without_sw)
    return remove_stop

def tokenizing(remove_stop):
    tokenizing = tokenizer.tokenize(remove_stop)
    return tokenizing

def convert_tokens(tokenizing):
    convert_tokens = tokenizer.convert_tokens_to_ids(tokenizing)
    return convert_tokens

#  Fungsi utk penggabungan semua fungsi preprocessing

def data_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens 
    if not word in stopwords.words(['english', 'portuguese', 'french', 'spanish', 'russian', 'italian', 'german'])]
    text = (" ").join(tokens_without_sw)
    tokenizing_bert = tokenizer.tokenize(text)
    text = tokenizer.convert_tokens_to_ids(tokenizing_bert)
    return text

def predict_sentiment(text):
    cleaned_text = data_cleaning(text) #pengaplikasian data cleaning

# Perform prediction
    prediction = model.predict([cleaned_text])
    prediction = float(prediction)
    return prediction

def main():
    st.title("Web App Sentimen Analisis")
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        with st.form(key='text_form'):
            input_text = st.text_area("Masukkan teks disini")
            submit_text = st.form_submit_button(label='Analisis')

        if submit_text:
            st.success("Hasil case folding:")
            case_folded = case_fold(input_text)
            st.write(case_folded)

            st.success("Hasil penghilangan tanda baca:")

            remove_punced = remove_punc(case_folded)
            st.write(remove_punced)
            
            st.success("Hasil penghilangan stopwords:")
            remove_stoped = remove_stop(remove_punced)
            st.write(remove_stoped)

            st.success("Hasil tokenisasi:")
            tokenizinged = tokenizing(remove_stoped)
            st.write(tokenizinged)

            st.success("Hasil konversi token:")
            convert_tokensed = convert_tokens(tokenizinged)
            st.write(convert_tokensed)

            st.success("Sentimen teks:")
            prediction = predict_sentiment(input_text)
            if prediction < 0.5:
                return st.write("Negatif (Produk tidak direkomendasikan)")
            else:
                return st.write("Positif (Produk direkomendasikan)")
        else:
            st.text ("Aplikasi ini dilatih menggunakan dataset review aplikasi spotify.")
            st.text("Metode yang digunakan adalah Bidirectional LSTM dan BERT Tokenizer.")


if __name__ == '__main__': 
    main()




