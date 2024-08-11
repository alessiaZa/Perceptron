import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import StandardScaler

#line equation: returns y = m*x + intercept
def f(x, m, q):
    return x * m + q

#create ground truth labels for points
#it applies f(x) to compute the labels
def createGroundTruthLabels(x, y, m, q):   
    labels = []
    for i in range(len(x)):
        label = 0
        if(y[i] > f(x[i], m, q)):
            label = 1

        labels.append(label)
    return labels

#create an array of dim random numbers in interval lower to higher
def randomArray(lower, higher, dim):
    a = []
    for i in range(dim):
        a.append(random.randint(lower, higher))
    return a


#parameters
numOfPoints = 2000
#line equation parameter
m = 4 
intercept = 0   #value of y for f(0) (point where the line intercept the y axis (x=0)

#determine plotter x and y axis range
xScale = []
yScale = []
if m == 0: #the line is parallel to the x axis  y = intercept
    xScale.append(0)
    xScale.append(1000)
    yScale.append(intercept-500)
    yScale.append(intercept+500)
else:
    xScale.append(0)
    xScale.append(1000)
    if(m < 0):  
        yScale.append(f(1000, m, intercept)) #negative m, swap lower with higher 
        yScale.append(f(0, m, intercept))
    
    else:
        yScale.append(f(0, m, intercept))   #positive m
        yScale.append(f(1000, m, intercept))

#create numPoints random x & y
x = np.array(randomArray(xScale[0], xScale[1], numOfPoints))
y = np.array(randomArray(yScale[0], yScale[1], numOfPoints))


#create ground truth labels for each x,y point
labels = createGroundTruthLabels(x, y, m, intercept)

# Initialize an empty plot
plt.figure(num="Dataset random generation")
#add axis labels and title to the plot
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Scattered points")

#plot scattered points
plt.scatter(x,y)

# Plot the linear function
xline = [xScale[0], xScale[1]]
yline = [f(xScale[0], m, intercept), f(xScale[1], m, intercept)]

plt.plot(xline, yline, color='red', linestyle='-', linewidth=2, label='Line')

# Add a legend
plt.legend()
#display the plot
plt.show()

#scale the values to get better training results
#standard scaling: z = (x - m)/d where z is the standardized value, 
#x the original value, m the mean value and d is the standard deviation
scaler = StandardScaler()
xy = np.column_stack((x, y))
scaled = scaler.fit_transform(xy)

#add labels column to the array
xyl = np.column_stack((scaled,labels))

#save dataset to file
np.savetxt("Perceptron/datam4q0.csv", xyl, delimiter=";")
