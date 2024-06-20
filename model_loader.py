import pickle
import tensorflow as tf

# Load sentiment analysis model
model_sentiment = tf.keras.models.load_model('weights/sentiment/sentiment_analysis.h5')
model_sentiment.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision')
    ]
)

# Load sentiment analysis tokenizer
with open('weights/sentiment/sentiment_analysis.pickle', 'rb') as f:
    tokenizer_sentiment = pickle.load(f)

# Load churn analysis model and label encoder
with open('weights/churn/churn_analysis.pickle', 'rb') as f:
    model_churn = pickle.load(f)

with open('weights/churn/label_encoder.pickle', 'rb') as f:
    label_encoder_churn = pickle.load(f)
