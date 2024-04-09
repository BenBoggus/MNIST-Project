import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from projectHelpers import *
import seaborn as sb
import matplotlib.pyplot as plt


M = np.genfromtxt(
    "C:/Users/benbo/CoolerCodeApp/Projects/Python/311Project2/data/mnist23-train.csv",
    delimiter=',', skip_header = True) #relative pathing was not working so I absolute pathed it instead

n, p = M.shape
print("Created MNIST Training array with:")
print("\t %g rows (images)" %n)
print("\t %g columns (labels and pixel intensities)\n" %p)


y = M[:,0]
X = M[:,1:]


V = getPCs(X,2)
Z = X @ V

#Problem 1
i1 = 0 #i1 is out initializor im using i1 because i dunno if ill need to use more loops and I dont wanna keep reusing i
while i1 < len(y):
    if y[i1] == 2:
        y[i1] = -1
    if y[i1] == 3:
        y[i1] = 1
    i1 +=1
beta, v = lsRegression(Z,y)


#Problem 2
f,k = lscPredict(Z,beta,v)
wrong = np.sum(np.sign(f[1:]) != np.sign(k[:-1]))
right = len(f) - wrong
percent_right = right / len(f) * 100
percent_wrong = wrong / len(f) * 100
print("\nPercentages Train")
print("%2s | %2s | %2s | %2s" % ("Right", "Wrong", "% Right", "% Wrong"))
print("="*33)
print("%.2d | %.2d | %.6f | %.6f" % (right, wrong, percent_right, percent_wrong))
print("\nConfusion Matrix Train")


#Problem 3
# Create dictionary storing the data 
# in the PC space and labels.
data_dict = {"x": Z[:,0],
        "y": Z[:,1],
        "Digits": f}

# Convert to a DataFrame
data = pd.DataFrame(data_dict)

plt.figure(figsize=(6,6))
sb.scatterplot(
    data = data, 
    x = "x", # x-coords.
    y = "y", # y-coords.
    hue="Digits", # colors.
    hue_norm=(min(f), max(f)),
    style = "Digits", # markers.
    s = 100, # marker size.
    alpha=0.25, # transparency.
    palette="coolwarm") # colormap.
plt.title(
    "Scatter plot of Digits in PC Space Training",
    fontsize = 16)                
plt.legend(
    title = "Digits", 
    title_fontsize = 14,
    fontsize = 14)
plt.xlabel("PC1 ($V_0$)")
plt.ylabel("PC2 ($V_1$)")
plt.show()


#Problem 4
def make_cm(predicted, actual):
    y_actual = pd.Series(actual, name = 'Actual')
    y_predicted = pd.Series(predicted, name = 'Predicted')
    return pd.crosstab(y_actual, y_predicted)
confusion = make_cm(f,y)
print(confusion)
plt.figure(figsize=(6,6))
sb.heatmap(confusion,annot=True)
plt.title("Heat Map of Training data",
          fontsize = 16)


#Problem 5
data_dict2 = {"Predicted Digits" : k,
              "Actual Digits" : y}

data2 = pd.DataFrame(data_dict2)

plt.figure(figsize = (6,6))
sb.histplot(data = data2,
            x = "Predicted Digits",
            hue = "Actual Digits",
            palette = "coolwarm",
            edgecolor = "w",)

plt.title("Histogram of Digits Training")
plt.xlabel("Predicted")
plt.ylabel("Frequency")
plt.show()








print("\nTesting Again With New Data\n")

#Problem 6
M_test = np.genfromtxt(
    "C:/Users/benbo/CoolerCodeApp/Projects/Python/311Project2/data/mnist23-test.csv",
    delimiter=',', skip_header = True) #relative pathing was not working so I absolute pathed it instead

n2, p2 = M_test.shape
print("Created MNIST Test array with:")
print("\t %g rows (images)" %n2)
print("\t %g columns (labels and pixel intensities)" %p2)
print()

# Split labels and images.
y_test = M_test[:,0] # labels.
X_test = M_test[:, 1:] # data matrix with images as row.

V2 = getPCs(X_test,2)
Z2 = X_test @ V2

#a) Regression
i2 = 0
while i2 < len(y_test):
    if y_test[i2] == 2:
        y_test[i2] = -1
    if y_test[i2] == 3:
        y_test[i2] = 1
    i2 +=1
beta2, v2 = lsRegression(Z2,y_test)

#b) Predict
f2, k2 = lscPredict(Z2,beta2,v2)
wrong2 = np.sum(np.sign(f2[1:]) != np.sign(k2[:-1]))
right2 = len(f2) - wrong2
percent_right2 = right2 / len(f2) * 100
percent_wrong2 = wrong2 / len(f2) * 100
print("\nPercentages Test")
print("%2s | %2s | %2s | %2s" % ("Right", "Wrong", "% Right", "% Wrong"))
print("="*33)
print("%.2d | %.2d | %.6f | %.6f" % (right2, wrong2, percent_right2, percent_wrong2))
print("\nConfusion Matrix Test")

# Create dictionary storing the data 
# in the PC space and labels.
data_dict3 = {"x": Z2[:,0],
        "y": Z2[:,1],
        "Digits": f2}

# Convert to a DataFrame
data3 = pd.DataFrame(data_dict3)

#c) 
plt.figure(figsize=(6,6))
sb.scatterplot(
    data = data3, 
    x = "x", # x-coords.
    y = "y", # y-coords.
    hue="Digits", # colors.
    hue_norm=(min(f2), max(f2)),
    style = "Digits", # markers.
    s = 100, # marker size.
    alpha=0.25, # transparency.
    palette="coolwarm"
    ) # colormap.
plt.title(
    "Scatter plot of Digits in PC Space Test",
    fontsize = 16)                
plt.legend(
    title = "Digits", 
    title_fontsize = 14,
    fontsize = 14)
plt.xlabel("PC1 ($V_0$)")
plt.ylabel("PC2 ($V_1$)")
plt.show()

#d) 
confusion2 = make_cm(f2,y_test)
print(confusion2)
plt.figure(figsize=(6,6))
sb.heatmap(confusion2, annot = True)
plt.title("Heatmap of Test Data",
          fontsize = 16)
plt.show()



data_dict4 = {"Predicted Digits" : k2,
              "Actual Digits" : y_test }
data4 = pd.DataFrame(data_dict4)
plt.figure(figsize=(6,6))
sb.histplot(
    data = data4,
    x = "Predicted Digits",
    hue = "Actual Digits",
    palette = "coolwarm",
    edgecolor = "w"
)
plt.title("Histogram of Predicted Test")
plt.xlabel("Predicted Test")
plt.ylabel("Frequency")
plt.show()

#Problem 7 Perform similarly? Has overfitting occured?
#a) Yes, the classifier performs similarly, there isn't a massive difference
#between the percentages of the actual and predicted between the two data sets used

#b) No, I don't think overfitting has occured, the classifier does make similar predicitions for both
#the training and the test data fed to it



#Problem 8 is in a different file since it's so long

