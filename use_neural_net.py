import os

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('mnist_neural_net.h5')


def predict_answer(sudoku):
    sudoku = np.array(sudoku)
    sudoku = sudoku / 9
    # Predicting
    sudoku = model.predict(np.expand_dims(sudoku, 0), verbose=0)
    return sudoku * 9


print(predict_answer(test))

'''
directory = os.fsencode("images")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_test = Image.open('images/' + filename)
    print(filename, predict_digit(image_test)[0])
'''