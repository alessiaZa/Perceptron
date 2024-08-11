import perc as perc
import matplotlib.pyplot as plt
import numpy as np

#hyperparameters
epochs = 52
learningRate = 0.01
n_inputs= 2

#instantiate the perceptron with two input (x and y)
p = perc.Perceptron(n_inputs, learningRate)

#load the dataset from disk and split the data between training and test
xyl = np.genfromtxt("Perceptron/datam2q10.csv", delimiter=";")
num_points = len(xyl)
num_train = int(num_points * 0.80)  #80% of data will be used for training, 20% for test
num_test = num_points - num_train

#split the array in training data and test data
x_train = np.array(xyl[:num_train,:2])
x_test = np.array(xyl[num_train:,:2])
y_train = np.array(xyl[:num_train,-1]).astype(int)
y_test = np.array(xyl[num_train:,-1]).astype(int)

#train the perceptron
print("start training")

errors, loss_per_epoch = p.train(x_train, y_train, epochs)

print("end training")


# Initialize an empty plot
plt.figure(num="Analisi grafica della loss function")
#add labels and title to the plot
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss per epoch")

n_epoch = [i for i in range(len(loss_per_epoch))]
plt.plot(n_epoch, loss_per_epoch, color="red")

# Display the plot
plt.show()

#test the confidence of the model

# Initialize an empty plot
plt.figure(num="Predizioni del modello")
#add labels and title to the plot
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Risultati dell'inferenza")
#plt.tight_layout()

#predict on a different set of points to test the performance of the model on new (unseen) set of data
print("start of prediction")
errorCount = 0

predictions = p.predict(x_test) #predicts on test data

print("end of prediction")

for i in range(len(predictions)):
    #if prediction is wrong 
    if predictions[i] != y_test[i]:
        errorCount += 1
        mycolor = 'red'
    else:
        mycolor = 'green' if predictions[i] == 0 else 'orange'   

    plt.scatter(x_test[i][0], x_test[i][1], color=mycolor)  # Plot each point with the color of the prediction


#print statistics
print(f"Total errors: {errorCount} out of {num_test}")
print(f"Percentage of error: {round(errorCount/num_test*100, 2)}%")
print(f"Percentage of success: {round(100 - errorCount/num_test*100, 2)}%")
print(f"final weights: {p.weights}")
print(f"final bias: {p.bias}")

# Display the plot
plt.show()