# Eye-for-Blind - Image Captioning using Attention Mechanism

WHO estimates show that there are approximately 285 million visually impaired people worldwide, out of which 39 million are completely blind. It is extremely difficult for them to carry out many daily activities, one of which is reading - from reading a newspaper / magazine to an important text message from your bank. 

Converting an image to text can be viewed as two sub tasks:- 
1. Generating cpations describing the image
2. Text to Speech Conversion.

A custom Deep Learning Model is built using Tensorflow and Keras based on Encoder - Decoder architecture with Bahdanau Attention Mechanism. The Model has been trained and tested on Flickr8k dataset. ImageNet, a CNN model pre-trained for image classification, is used for feature extraction. These features are passed through an RNN , which generates a caption , with due regard to previously generated words. This model improves the standard encoder-decoder architecture by the addition of an attention mechanism which helps the network focus on a part of the image (instead of focusing on the entire image all the time ) based on the words previously generated. The generated caption is ranked based on Greedy Search and evaluated using the Bilingual Evaluation Understudy ( BLEU ) score. Neural Architecture in this project is based on Show, Attend & Tell,2015.

Libraries : Keras, Tensorflow

Concepts : Feature Extraction using ImageNet, GRU, Bahdanau Attention Mechanism, Greedy Search, Blue Score
