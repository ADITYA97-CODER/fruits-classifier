import cv2
import numpy as np
from keras.models import load_model
model = load_model('fruit.h5')
cap = cv2.VideoCapture(0)
name = ['apple','bannana','mixed','orange']
def preprocess(img):
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY )
    img = cv2.equalizeHist(img)
    img= img/255
    return img

while True:
    success , image = cap.read()
    imag = cv2.resize(image , (32,32))
    imag = preprocess(imag)
    imag = np.asarray(imag)
    imag = imag.reshape(1,32,32,1)
    no = np.argmax(model.predict(imag))

    cv2.putText(image,str(name[no]),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.imshow("image",image)
    cv2.waitKey(1)