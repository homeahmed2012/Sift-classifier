
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




       
























# nearest=[]
# #Match descriptors.
# matches =[]
# faces_distance =[]

# for i in range(0, len(listOfDesc)):
#      for j in range(i, len(listOfDesc)):
#           matches = bf.match(listOfDesc[i],listOfDesc[j])
#           distance = get_distance(matches)
#           faces_distance.append([imageNames[i], imageNames[j],distance])
#           if distance < 100:
#                print imageNames[i], imageNames[j],distance

# save_result(faces_distance)
# print Faces_distance

# def get_distance(matches):
#      # Sort them in the order of their distance.
#      matches = sorted(matches, key = lambda x:x.distance)
#      sum = 0
#      n = len(matches) # return number of matched descriptors
     
#      if n > 20:
#           n = 20
#      for i in range(1,n):
#           #print "matches", mat.distance
#           sum +=matches[i].distance
#      return sum/n


# def save_result(result):

#      # Copy the results to a pandas dataframe with an "id" column and
#      output = pd.DataFrame( data=result )
#      # Use pandas to write the comma-separated output file
#      output.to_csv( "f:/face_matching_model.csv", index=False, quoting=3)
     
     
     
     
     
     
     
     



# read the image 
#img = cv2.imread('home.jpg')

# convert to gray scale
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
# create sift object
#sift = cv2.xfeatures2d.SIFT_create()

# get sift key points 
#kp = sift.detect(gray,None)

# to draw key points on the image
#img = cv2.drawKeypoints(gray,kp, img)
#cv2.imshow('img', img)

#kp, des = sift.detectAndCompute(gray,None)