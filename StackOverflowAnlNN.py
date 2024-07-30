import tensorflow as tf
import pandas as pd
import numpy as np
import re
from keras import layers

#NOTE This code will train a classification AI and overwrite the file SackOverflowClassificationAI

class model_package:
    def __init__(self, model: tf.keras.Model, loss: float, accuracy: float):
        self.model = model
        self.loss = loss
        self.accuracy = accuracy
        
raw_train_datset = tf.keras.preprocessing.text_dataset_from_directory("./StackOverflow/train", label_mode="categorical", validation_split=0.2,subset= "training", batch_size=32, seed=42)
raw_validation_dataset = tf.keras.preprocessing.text_dataset_from_directory("./StackOverflow/train", label_mode="categorical", validation_split=0.2,subset= "validation", batch_size=32, seed=42)
raw_test_dataset = tf.keras.preprocessing.text_dataset_from_directory("./StackOverflow/test", label_mode="categorical", batch_size=32, seed=42)

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_train_datset.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_datset.map(vectorize_text)
validation_ds = raw_validation_dataset.map(vectorize_text)
test_ds = raw_test_dataset.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16


model_numbers = 20
model_collection = []
for i in range(model_numbers):

    model = tf.keras.models.Sequential([
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(20, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(4, activation='relu'),
        layers.Softmax()
    ])

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=[tf.metrics.CategoricalAccuracy()]
    )

    model.fit(
        train_ds,
        validation_data= validation_ds,
        epochs=10
    )

    loss, accuracy = model.evaluate(test_ds)
    print("\n", "Model number " + str((i + 1)),"The model loss is: " + str(loss), "The model accuracy is: " + str(accuracy), '\n')
    model_to_add = model_package(model, loss=loss, accuracy=accuracy)
    model_collection.append(model_to_add)

bubbled_max = 0
for package in model_collection:
    if package.accuracy >= bubbled_max:
        bubbled_max = package.accuracy
        model = package.model

loss, accuracy = model.evaluate(test_ds)
print("\n", "Final model,", "The model loss is: " + str(loss), "The model accuracy is: " + str(accuracy), '\n')

predictior = tf.keras.models.Sequential([vectorize_layer, model])
predictior.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=[tf.metrics.CategoricalAccuracy()]
)
prediction = predictior.predict(["how to check object names in loops i have 31 numericipdowns (nup1,nup2,nup3....nup31), i want to add this numericupdowns values in datagridview.i used ""for"" loop and ""switch"", now i want to make something like this:..for(int i=1;i&lt;32;i==){.   if(nup+i.value&gt;0){.   datagridview1.rows.add((nup+i).tostring(0).   }.}...anybody can to help me?"])
print(str(prediction))

predictior.save(".\Misc\TestModels\ModelFiles\SentimentAnlNN_1\StackOverflowSnAnl\SackOverflowClassificationAI")