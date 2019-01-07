import json
import numpy as np

train_labels = np.load("train_labels.npy")
train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")

size_train_set = len(train_data)
size_test_set = len(test_data)
num_words = len(train_data[0])

learning_rate = 0.01

weights = np.random.rand(num_words)  # will be updated
for paper_number in range(size_train_set):
    doc = train_data[paper_number]
    # get a prediction for this paper
    result = np.dot(doc, weights)
    if result > 0:
        prediction = 1
    else:
        prediction = 0
    correct = train_labels[paper_number]
    error = correct - prediction
    #update the weights
    if error != 0:
        for ind in range(len(weights)):
            weights[ind] = weights[ind]

print("Done.")