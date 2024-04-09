import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from projectHelpers import *
import sklearn
import seaborn as sb
import matplotlib.pyplot as plt
#Confusion Matrix funciton
def make_cm(predicted, actual):
    y_actual = pd.Series(actual, name = 'Actual')
    y_predicted = pd.Series(predicted, name = 'Predicted')
    return pd.crosstab(y_actual, y_predicted)


M = np.genfromtxt("C:/Users/benbo/CoolerCodeApp/Projects/Python/311Project2/data/fashion-mnist_67_train.csv", delimiter=",", skip_header=True)
y = M[:,0] # labels.
X = M[:, 1:] # data matrix with images as rows.
n, p = M.shape
print("Created MNIST Training array with:")
print("\t %g rows (images)" %n)
print("\t %g columns (labels and pixel intensities)\n" %p)

# Calculate PC-vectors.
V = getPCs(X, 2)
# Project onto the PC-space.
Z = X @ V 

M_test = np.genfromtxt("C:/Users/benbo/CoolerCodeApp/Projects/Python/311Project2/data/fashion-mnist_67_test.csv", delimiter=",", skip_header=True)
y_test = M_test[:,0]
X_test = M_test[:,1:]
l, j = M_test.shape
print("Created MNIST Test array with:")
print("\t %g rows (images)" %l)
print("\t %g columns (labels and pixel intensities)\n" %j)
V_test = getPCs(X,2)
Z_test = X_test @ V_test

#a) Regression for training data
i3 = 0
while i3 < len(y):
    if y[i3] == 6:
        y[i3] = -1
    if y[i3] == 7:
        y[i3] = 1
    i3+=1
beta_fash, v_fash = lsRegression(Z,y)

#Regression for testing data
i4 = 0
while i4 < len(y_test):
    if y_test[i4] == 6:
        y_test[i4] = -1
    if y_test[i4] == 7:
        y_test[i4] = 1
    i4 += 1
beta_flash_test, v_flash_test = lsRegression(Z_test, y_test)

#b) Predictions and Confusion Matrix for training data
print("Confusion Matrix for training\n")
f_fash, k_fash = lscPredict(Z,beta_fash,v_fash)
confusion_fash = make_cm(f_fash, y)
wrong_f = np.sum(np.sign(f_fash[1:]) != np.sign(k_fash[:-1]))
right_f = len(f_fash) - wrong_f
percent_right_f = right_f / len(f_fash) * 100
percent_wrong_f = wrong_f / len(f_fash) * 100
print("%2s | %2s | %2s | %2s" % ("Right", "Wrong", "% Right", "% Wrong"))
print("="*33)
print("%.2d | %.2d | %.6f | %.6f" % (right_f, wrong_f, percent_right_f, percent_wrong_f))
print("\nMatrix for Training")
print(confusion_fash)

print("\nUsing Test Data\n")

#Predictions and Confusion for test data
print("Confusion Matrix for Test\n")
f_fash_test, k_fash_test = lscPredict(Z_test, beta_flash_test, v_flash_test)
confusion_test = make_cm(f_fash_test, y_test)
wrong_test = np.sum(np.sign(f_fash_test[1:]) != np.sign(k_fash_test[:-1]))
right_test = len(f_fash_test) - wrong_test
percent_right_fash_test = right_test / len(f_fash_test) * 100
percent_wrong_fash_test = wrong_test / len(f_fash_test) * 100
print("%2s | %2s | %2s | %2s" % ("Right", "Wrong", "% Right", "% Wrong"))
print("="*33)
print("%.2d | %.2d | %.6f | %.6f" % (right_test, wrong_test, percent_right_fash_test, percent_wrong_fash_test))
print("\nMatrix for Test")
print(confusion_test)

#Scatter Plots for Training/Test data
#Data Table for Training
data = {"x": Z[:,0],
        "y": Z[:,1],
        "Type": f_fash}
df = pd.DataFrame(data)

#Data Table for Test
data_test = {"x":Z_test[:,0],
             "y":Z_test[:,1],
             "Type": f_fash_test}
df_test = pd.DataFrame(data_test)

fig = plt.figure(figsize=(8,8))
ax = sb.scatterplot(data = df, x = "x", y = "y", 
                hue= "Type",
                hue_norm=(6,7),
                alpha=0.5,
                style = "Type",
                s = 250,
                palette="coolwarm")
plt.title(
    "Scatter plot of Shirts and Shoes from Fashion MNIST Training", 
    fontsize = 16)                
plt.xlabel("PC1", fontsize = 14)
plt.ylabel("PC2", fontsize = 14)
plt.legend(
    title = "Garment", 
    title_fontsize = 14,
    loc = "lower right", fontsize = 14
)
plt.show()

#Scatter for Test Data
fig2 = plt.figure(figsize=(8,8))
ax2 = sb.scatterplot(data = df_test, x = "x", y = "y", 
                hue= "Type",
                hue_norm=(6,7),
                alpha=0.5,
                style = "Type",
                s = 250,
                palette="coolwarm")
plt.title(
    "Scatter plot of Shirts and Shoes from Fashion MNIST Test", 
    fontsize = 16)                
plt.xlabel("PC1", fontsize = 14)
plt.ylabel("PC2", fontsize = 14)
plt.legend(
    title = "Garment", 
    title_fontsize = 14,
    loc = "lower right", fontsize = 14
)
plt.show()


#Heat Maps for Training/Test Data

#Training Data
fig3 = plt.figure(figsize = (6,6))
sb.heatmap(confusion_fash,annot=True)
plt.title("Heatmap for Training Data",
          fontsize = 16)
plt.show

#Testing Data
fig4 = plt.figure(figsize = (6,6))
sb.heatmap(confusion_test,annot=True)
plt.title("Heatmap for Test Data",
         fontsize = 16)
plt.show()

#HistPlots for Training/Test Data

#Training Data
data2_train = {"Predicted" : k_fash,
               "Actual" : y}

df2 = pd.DataFrame(data2_train)
fig5 = plt.figure(figsize = (6,6))
sb.histplot(
    data = df2, 
    x = "Predicted",
    hue = "Actual",
    palette = "coolwarm",
    edgecolor = "w"
)
plt.title("HistPlot Train")
plt.xlabel("Predicted")
plt.ylabel("Frequency")
plt.show


data_test_2 = {"Predicted" : k_fash_test,
               "Actual": y_test}

df_test2 = pd.DataFrame(data_test_2)
fig6 = plt.figure(figsize = (6,6))
sb.histplot(
    data = df_test2,
    x = "Predicted",
    hue = "Actual",
    palette = "coolwarm",
    edgecolor = "w"
)
plt.title("Histplot Test")
plt.xlabel("Predicted")
plt.ylabel("Frequency")
plt.show()



#Question in Problem 8
#The classified does not appear to be overfitting the results of both the training and the testing data are
#very similar

