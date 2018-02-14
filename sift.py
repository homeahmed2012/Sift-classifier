
print('running...............please wait')

# importing the libraries 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# the path to the folder that contains the training images
train_path = 'train'

# list of tubles containing the image name and number of descriptors 
imageNames=[]

# list of all descriptors of all images
listOfDesc=[]

# Initiate SIFT
sift = cv2.xfeatures2d.SIFT_create()

# loop on all images and extract thier descriptors
for path, subdirs, files in os.walk(train_path):
    for filename in files:
        f = os.path.join(path, filename)
        
        # read the image
        img1 = cv2.imread(f,0)
        
        # extract the key points and the descriptors
        kp, des = sift.detectAndCompute(img1,None)
        listOfDesc.extend(des) # add image descriptors (i.e. corners)
        filename = filename.split('.', 1 )[0]
        imageNames.append((filename, len(des)))
         
         

# numpy array of all the descriptors
X = np.array(listOfDesc)

# number of K-mean clusters
K = 40

# Using k mean to classify all the descriptors
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = K, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

result = np.zeros((len(imageNames), K))

names = []
for name in imageNames:
     names.append(name[0])

names = np.array(names)     

# for each image find the number of descriptors in each class      
ind = 0
for i in range(len(imageNames)):
     img = imageNames[i]
     
     for j in range(img[1]):
          result[i][y_kmeans[j+ind]] += 1
          
     ind += img[1]


# the path to the folder that contains the test images
test_path = 'test'
test_img_name = input('please enter the name of the image from test folder\n') 

# read the test image
test_img = cv2.imread(os.path.join(test_path, test_img_name),0)

# extract key points and descriptors form the test image
kp, des = sift.detectAndCompute(test_img,None)


# classify the test image descriptors
y_test_pred = kmeans.predict(np.array(des))

# vector continas the number of test image descriptor in each class
img_vec = np.zeros((1, 40))

for t in y_test_pred:
     img_vec[0][t] += 1 


# Feature Scaling before applying K-NN classification
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(result)
X_test = sc.transform(img_vec)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, names)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

print(y_pred[0])



