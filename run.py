import cv2
import numpy as np
import dlib
from facepoints import extremepoints
from collections import deque
import argparse
from img_pyramids import merge


def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    #print len(channels)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def meann(deq):
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for i in deq:
        x1.append(i[0])
        y1.append(i[1])
        x2.append(i[2])
        y2.append(i[3])
    return (int(sum(x1)/len(x1)),int(sum(y1)/len(x1)),int(sum(x2)/len(y2)),int(sum(y2)/len(x1)))

def runner():


    obj=extremepoints()
    obj2=merge()
    deq=deque(maxlen=10)
    cap=cv2.VideoCapture(args.input)
    ret,frame=cap.read()
    h,w,c=frame.shape
    aspect=w/h
    h=300
    w=int(aspect*h)
    out=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (w,h))
    while ret:
        img=cv2.imread("temp/img.png",1)
        mask=cv2.imread("temp/mask.png",0)
        ret,frame=cap.read()
        #mask2=np.zeros((h,w,3))
        frame=cv2.resize(frame,(w,h))

        p1,p2=obj.getCoordinates(frame)
        x1,y1=p1
        x2,y2=p2
        deq.append((x1,y1,x2,y2))
        x1,y1,x2,y2=meann(deq)
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        width=x2-x1
        height=y2-y1
        img=cv2.resize(img,(width,height))
        mask=cv2.resize(mask,(width,height))
        mask_inv=cv2.bitwise_not(mask)
        background=frame[y1:y2,x1:x2]

        onlybackground=np.float32(cv2.bitwise_and(background,background,mask=mask_inv))/255.0
        onlyforeground=np.float32(cv2.bitwise_and(img,img,mask=mask))/255.0
        mask=np.float32(mask)/255.0

        mask=mask.reshape((height,width,1))
        combined=mask*onlyforeground+onlybackground*(1-mask)
        combined=np.uint8(255*combined)
        mask=np.uint8(mask*255)
        h2,w2,_=mask.shape
        output = cv2.seamlessClone(combined, background, mask, (int(w2/2),int(h2/2)), cv2.NORMAL_CLONE)

        frame[y1:y2,x1:x2]=output
        #frame=obj2.merg(frame)
        #frame=hisEqulColor(frame)
        out.write(frame)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break

    out.release()
    cv2.destroyAllWindows()
    cap.release()

if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-img","--image",help="give path to your photo")
    args=parser.parse_args()
    runner(args)
