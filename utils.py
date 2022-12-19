import io
import os
import glob
import time
import pickle

import numpy as np
import tensorflow as tf
from gtts import gTTS

#Hyper Parameters
embed_dims = 256
units = 512
vocab_size = 5001 # top 5000 words + 1 for <unk>
max_length_caption = 38 #Got from EDA
attention_ip_features_shape = 64 #8*8 squeezed to 64


def preprocess_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299,299))
    img = tf.keras.applications.inception_v3.preprocess_input(img) #Normalize the image within the range of -1 to 1
    return img, img_path


#Encoder, Attention & Decoder Classes
class Encoder(tf.keras.Model):
    def __init__(self,embed_dims=256):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dims, activation='relu') # Shape of feature :(64,2048) --> (64,256) to concatenate with embedded vector for word of caption
    
    def call(self,features):
        features = self.dense(features)
        return features



class Attention(tf.keras.Model):
    def __init__(self,units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) #Dense layer for linear transforming encoder features: (64,256)-->(64,512)
        self.W2 = tf.keras.layers.Dense(units) #Dense layer for linear transforming embedded vector of word : (1,256)-->(1,512)
        self.V = tf.keras.layers.Dense(1) #Dense layer with unit 1 for input of tanh(W1 + W2)
        self.units = units

    def call(self,features,hidden_state):
        # features shape - (batch_size, 64, embed_dims), hidden_state shape - (batch_size, units=512)
        hidden_with_time_axis = tf.expand_dims(hidden_state,1) # (batch_size, 1, units=512)
        #When adding two input values of different shapes, "Add" follows NumPy broadcasting rules
        attention_hidden_layer = tf.keras.activations.tanh( self.W1(features) + self.W2(hidden_with_time_axis) ) # (batch_size, 64,units=512 ) + (batch_size, 1, units=512) --> (batch_size, 64, units=512) --> Ex. A = [1,2,3,4,5], B = [1], A+ B = [2,3,4,5,6]
        score = self.V(attention_hidden_layer) # (batch_size, 8*8, 1)
        attention_weights = tf.keras.activations.softmax(score, axis=1) # attention_weights shape: (batch_size, 8*8, 1)
        context_vector = attention_weights * features # context_vector shape - (batch_size, 8*8, embed_dims)
        context_vector = tf.reduce_sum(context_vector, axis = 1) # Reduce shape to (batch_size, embed_dims)
       
        return context_vector, attention_weights



class Decoder(tf.keras.Model):
    def __init__(self, embed_dims, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention(self.units) #iniitalizing Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dims) #building an Embedding layer
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #Dense layer
        
    def call(self, x, features, hidden_state):
        # context_vector shape - (batch_size, embed_dims), attention_weights shape - (batch_size, 8*8, 1)
        context_vector, attention_weights = self.attention(features, hidden_state)
        embed = self.embed(x) # (batch_size, 1) --> (batch_size,1, embed_dims)
        concated_input = tf.concat([tf.expand_dims(context_vector,1), embed], axis=-1) # (batch_size, 1, embed_dims+embed_dims )
        # print("concated_input shape",concated_input.shape)
        # print("embed shape",embed.shape)
        # print("context_vector shape",context_vector.shape)
        output, hidden_state = self.gru(concated_input) # Output Shape - (batch_size, max_length_caption, units=512)  as we have set return_sequences=True, hidden_state shape - (batch_size, units=512) that's why we expand dimn to (batch_size, 1 , units=512) in attention model
        # print("GRU process completed")
        output = self.d1(output)
        # print("Before reshape output shape", output.shape)
        # print("hidden_state shape", hidden_state.shape)
        output = tf.reshape(output,(-1,output.shape[2]))   # o/p shape- (batch_size*max_length_caption, units=512)
        # print("after reshape output shape", output.shape)
        output = self.d2(output) # o/p shape- (batch_size*max_length_caption, vocab_size=5001)
        
        return output, hidden_state, attention_weights


    def init_state(self,batch_size):
        return tf.zeros((batch_size, self.units)) 



def get_tokenizer(tokenizer_path = "./checkpoint/tokenizer.pkl"):
    with open(tokenizer_path, "rb") as tokenizer:
        return pickle.load(tokenizer)
    

def load_model(checkpoint_path = './checkpoint/train'):
    tokenizer = get_tokenizer()
    img_model =tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = img_model.input
    new_output = img_model.layers[-1].output
    img_features_extract_model = tf.keras.Model(new_input,new_output)
    
    encoder = Encoder(embed_dims)
    decoder = Decoder(embed_dims, units, vocab_size)
    optimizer = tf.keras.optimizers.Adam()
    
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    
    return encoder, decoder, tokenizer, img_features_extract_model
    

#Load model    
encoder, decoder, tokenizer, img_features_extract_model = load_model()


def filter_text(text):
    filters = ['<start>', '<unk>', '<end>']
    text_split = text.split()
    [text_split.remove(j) for k in filters for j in text_split if j == k]
    text_join = ' '.join(text_split)
    return text_join



def predict_caption(img_path):
    preprocessed_img = tf.expand_dims(preprocess_img(img_path)[0],0) # preprocessed_img shape = (1,1,1,3)
    extracted_features = img_features_extract_model(preprocessed_img) # Extracted features Shape = (1,8,8,2048)
    reshaped_features = tf.reshape(extracted_features,(extracted_features.shape[0],-1, extracted_features.shape[3])) # Reshaped features Shape = (1,64,2048)
   
    encoded_features = encoder(reshaped_features) # Shape =(1,64,256) --> 256=embed_dims

    decoder_ip = tf.expand_dims([tokenizer.word_index['<start>']], 1) #shape =(1,1)
    print("decoder_ip Shape", decoder_ip.shape)
    hidden_state = decoder.init_state(batch_size=1) #shape=(1,512) --> 512 = units
    pred_words = []
    attention_plot = np.zeros((max_length_caption, attention_ip_features_shape))
  
    for i in range(max_length_caption):
        predictions, hidden_state, attention_weights = decoder(decoder_ip,encoded_features,hidden_state) # predictions = (1,5001), attention_weights = (1,64)
        attention_plot[i] = tf.reshape(attention_weights,(-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        word = tokenizer.index_word[predicted_id]
        pred_words.append(word)

        if word =='<end>':
            pred_caption = ' '.join(pred_words).rsplit(' ', 1)[0]
            return pred_caption
        decoder_ip =  tf.expand_dims([predicted_id], 0)

    return pred_caption.rsplit(' ', 1)[0]


def text_to_audio(text, output_language='en'):
    tts = gTTS(text, lang=output_language, slow=False, tld="com")
    try:
        audio_file_name = text[0]
    except:
        audio_file_name = "audio"
    tts.save(f"./temp/audios/{audio_file_name}.mp3")
    print("audio file saved")
    return audio_file_name


