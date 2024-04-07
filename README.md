# Image_Caption_Generator

Project Information:
The objective of the project is to predict the captions for the input image. The dataset consists of 8k images and 5 captions for each image. The features are extracted from both the image and the text captions for input. The features will be concatenated to predict the next word of the caption. CNN is used for image and LSTM is used for text. BLEU Score is used as a metric to evaluate the performance of the trained model.

dataset link: https://www.kaggle.com/adityajn105/flickr8k


Libraries Used: 
numpy,
matplotlib,
keras,
tensorflow,
nltk.

Neural Networks Used: 
VGG16 Network,
CNN-LSTM Network.


For This Project To Work On Your PC, Download the Code and the user will need to make a diectory for saving the best model after training (in the folder named *working*, you can change the path according to your needs.)
