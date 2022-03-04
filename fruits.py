import  os
import cv2
from keras import activations
from keras.backend import flatten
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
list = os.listdir('train')
images = []
name =[]
nam = str(list[3])

def split(names):
    return [char for char in names]
for y in list:
    n = []
    s=''
    split_name  = split(y)
    for z in split_name:
        if z=='_':
            break
        else :
            n.append(z)
    n=''.join(n)
    name.append(n)

name = np.asarray(name)



def preprocess(img):
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY )
    img = cv2.equalizeHist(img)
    img= img/255
    return img
for x in list:
    img = cv2.imread('train/'+x)
    img = cv2.resize(img,(32,32))
    img= preprocess(img)
    images.append(img)
   
images = np.asarray(images)
print(images.shape)
print(name.shape)
images  = np.reshape(images,(images.shape[0],images.shape[1],images.shape[2],1))
print(images.shape)

le= LabelEncoder()
le.fit(name)
name = le.transform(name)
print(le.classes_)
#name = to_categorical(name)
'''
def create_model():
    model = Sequential()
    nf1 = 100
    nf2 = 50
    nf3 = 40
    ns1 = (3,3)
    ns2 = (2,2)
    ps = (2,2)
    noofnode = 500
    model.add((Conv2D(nf1,ns1,input_shape = (32,32,1),activation = 'relu')))
    model.add((Conv2D(nf1,ns1,activation = 'relu')))
    model.add(MaxPooling2D(pool_size = ps))
    model.add((Conv2D(nf2,ns2 , activation= 'relu',)))
    model.add((Conv2D(nf3,ns2,activation = 'relu')))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noofnode,activation = 'relu'))
    model.add(Dense(200,activation = 'relu'))
    model.add(Dense(4,activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss= 'sparse_categorical_crossentropy' ,metrics= ['accuracy'])
    return model
model = create_model()
datagen = ImageDataGenerator( width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
history  = model.fit(datagen.flow(images,name,batch_size  = 100), epochs = 30,shuffle = 1 )
model.save('fruit.h5')
'''