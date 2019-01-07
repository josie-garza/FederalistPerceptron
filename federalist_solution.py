import json
import numpy as np

train_labels = np.load("train_labels.npy")
train_data = np.load("train_data.npy")
test_data = np.load("test_data.npy")

size_train_set = len(train_data)
size_test_set = len(test_data)
num_words = len(train_data[0])

w = np.random.rand(num_words)

print("Starting training")

unit_step = lambda x: 0 if x < 0 else 1
learning_rate = 0.01

for i in range(50):
    wrong = 0
    correct = 0 
    for x in range(size_train_set):
        result = np.dot(w, train_data[x])
        error = train_labels[x] - unit_step(result)
        float_error = train_labels[x] - result
        if error == 0:
            correct += 1
        else:
            wrong += 1
        w += learning_rate * error * train_data[x]
    print("For round %d, we got %d right and %d wrong"%(i,correct, wrong))

for i in range(size_test_set):
    label = unit_step(np.dot(w, test_data[i]))
    if label == 0 :
        print("I think the %dth unknown paper is by Hamilton"% i)
    else:
        print("I think the %dth unknown paper is by Madison"% i)




