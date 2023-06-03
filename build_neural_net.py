import tensorflow as tf

x_train, x_test = x_train / 255.0, x_test / 255.0

# define the structure of the model using keras (part of tensorflow used to build NN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(9, 9)),  # reformat a 2D grid into a 1D list of length 28 x 28
    tf.keras.layers.Dense(128, activation='relu'),  # 128 = number of neurons, each of which recognises a different feature and returns how strong that feature is (0->1). Dense means that every entry in the 28x28 array maps to every neuron. A smaller number of neurons would make a less accurate, but faster and more stable model
    tf.keras.layers.Dropout(0.2),  # randomly removes 20% of outputs to avoid the model having only 1 way of identifying the images, preventing over-fitting (so it learns the features not the thing as a whole)
    tf.keras.layers.Dense(81)   # returns the probability that it is each of numbers 0-9
])
print(model.summary())

# define loss function we want for the model - this helps to optimise the performance of the model - it incurs a penalty when it gets the output wrong as it loops through the training data - this can be sophisticated to for example give a higher penalty for numbers further away (e.g. if using for directions)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# define how we want the model to train itself
model.compile(optimizer='adam',  # different mathematical approaches to optimising
              loss=loss_fn,  # helps to optimise the performance of the model - see above
              metrics=['accuracy'])  # defines what we want the model to optimise for - in this case accuracy but could be precision etc. see https://keras.io/api/metrics/

# fits the  model - epoch = number of times the NN goes through the data to try to fit it; higher number = more accurate but slower
model.fit(x_train, y_train, epochs=10)

# assess the accuracy of the model using the test data, verbose = amount of information shared with a higher number providing more info
evaluation = model.evaluate(x_test, y_test, verbose=2)
print(evaluation)

# save the settings of the trained NN, so we can use it for predictions
model.save('sudoku_neural_net.h5')
