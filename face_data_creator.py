import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import sys
import os
import shutil

''' 
    This takes all images in image_to_be_added in data folder
    Then it detects faces in each each image and ask their name
    It then transform image to np array and save it to database

'''
 
# classifier to identify face in image using opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

if sys.argv[1] == 'update':
    print("Loading existing data")
    data= np.load('data.npy') 
    print(data.shape)
    

elif sys.argv[1]=='new' and input("type 'yes' to confirm :")=='yes':
    # renaming data to previous data
    if os.path.isfile("data.npy"): 
        print('Removing existing data')
        if os.path.isfile("prev_data.npy"):
            os.remove("prev_data.npy")

        os.rename("data.npy","prev_data.npy")
        print("---Data cleaned")

    # deleting saved database faces
    print("removing all saved faces")
    faces = glob.glob('face/*')
    for j in faces:
        os.remove(j)

    # taking care of previous db images
    print("moving images to unused folder")
    added = glob.glob('added_images/*')

    for j in added:
        try:
            shitil.move(j,'unused_images/')
        except:
            print('already')
            os.remove(j)
        
    # creating new databse
    data= np.empty(150*150+1)
    print(data.shape)

else:
    sys.exit()
    





# convering png image in to jpg format
pngs = glob.glob('to_be_added_images/*.png')
for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img) 
    os.remove(j)
print("Name the face 'none' if not to include image in database") 
train_images = glob.glob("to_be_added_images/*.jpg")
for image in train_images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detecting all faces in image..
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi_color = img[y:y+h, x:x+w]

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (150, 150), interpolation=cv2.INTER_AREA)
        print(roi_gray.shape)
        plt.imshow(roi_color, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
       
        
        name =input('type his name:no space:')
        if name != 'none':
            roi_data = roi_gray.ravel()
            roi_data =np.hstack((roi_data,np.array([name])))
            data=np.vstack((data,roi_data))
    

            file_name = name + str(x)+str(w)+str(y) + '.png'
            cv2.imwrite('face/'+ file_name,roi_gray)

    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.show()
    try:
        shutil.move(image,'added_images/')
    except:
        print("already")
        os.remove(image)

print("removing none lebeled images")
print(data.shape)
np.save('data',data)


