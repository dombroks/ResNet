import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np


def initialize_model():
    # Initialize the model
    return tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000
    )


def preprocess_image(image_sample):
    # Convert image into array
    transformed_image = image.img_to_array(image_sample)

    # Expand the dimension
    transformed_image = np.expand_dims(transformed_image, axis=0)

    # Preprocess the image
    transformed_image = preprocess_input(transformed_image)

    return transformed_image


def decode_prediction(prediction):
    prediction_label = decode_predictions(prediction, top=1)
    return prediction_label[0][0][1], prediction_label[0][0][2] * 100


def predict(image1):
    # Load the model
    model = initialize_model()

    # Process image
    transformed_image = preprocess_image(image1)

    # Predict and print the result
    prediction = model.predict(transformed_image)
    decoded_prediction, score = decode_prediction(prediction)
    print('%s (%.2f%%)' % (decoded_prediction, score))
    
    return decoded_prediction, score
