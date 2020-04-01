import argparse
import tensorflow as tf
import cv2
from keras.models import load_model
import numpy as np
import dlib
from facepoints import extremepoints


def smooth(e1,e2,mask):
    val=(mask-e1)/(e2-e1)
    x=np.clip(val,0.0,1.0)
    return x*x*(3-2*x)

def get_images(args):

    obj=extremepoints()
    img=cv2.imread(args.image,1)
    h,w,_=img.shape
    aspect=float(w)/h
    h=500
    w=int(500*aspect)
    frame=img.copy()
    frame=cv2.resize(frame,(w,h))
    p1,p2=obj.getCoordinates(frame)
    x1,y1=p1
    x2,y2=p2
    img=cv2.resize(img,(128,128))/255.0
    model=load_model("trained_model/deconv_bnoptimized_munet.h5")
    mask=model.predict(img.reshape((1,128,128,3)))
    mask=np.reshape(mask,(128,128,1))
    mask=mask*255
    mask=np.uint8(mask)
    mask=cv2.GaussianBlur(mask,(5,5),0)
    mask=cv2.dilate(mask,None,iterations=1)
    mask=cv2.erode(mask,None,iterations=1)
    t,mask=cv2.threshold(mask,120,255,cv2.THRESH_BINARY)
    mask=cv2.resize(mask,(w,h))
    cropped_img=frame[y1:y2,x1:x2]
    cropped_mask=mask[y1:y2,x1:x2]
    c2=np.float32(cropped_mask)/255.0
    cropped_mask=smooth(0.3,0.5,c2)
    cropped_mask=np.uint8(cropped_mask*255)
    cropped_mask=cv2.erode(cropped_mask,None,iterations=2)
    png=cv2.bitwise_and(cropped_img,cropped_img,mask=cropped_mask)
    cv2.imwrite("temp/img.png",png)
    cv2.imwrite("temp/mask.png",cropped_mask)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-img","--image",help="give path to your photo")
    args=parser.parse_args()
    get_images(args)
