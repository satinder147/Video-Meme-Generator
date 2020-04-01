import cv2
import dlib

class extremepoints:

    
    def __init__(self):
        self.detector=dlib.get_frontal_face_detector()
        self.predictor=dlib.shape_predictor("trained_model/land.dat")
    
    def getCoordinates(self,frame):
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        rects=self.detector(grey,1)
        for rect in rects:
            x1,y1,x2,y2=rect.left(),rect.top(),rect.right(),rect.bottom()
            landmarks=self.predictor(grey,rect)
            bottom=(landmarks.part(8).x,landmarks.part(8).y)
            d=landmarks.part(54).y-landmarks.part(27).y
            top=(landmarks.part(8).x,landmarks.part(8).y-int(d/(0.36)))
            left=landmarks.part(0).x,landmarks.part(0).y
            right=landmarks.part(16).x,landmarks.part(16).y

            return (int(left[0]*0.85),int(top[1]*0.8)),(int(right[0]*1.1),bottom[1])
        return (0,0),(0,0)

if __name__=="__main__":
    cap=cv2.VideoCapture(0)
    obj=extremepoints()
    while(1):
        frame=cv2.imread("img.jpg",1)
        p1,p2=obj.getCoordinates(frame)
        cv2.rectangle(frame,p1,p2,(0,255,0),2)
        cv2.imshow("f",frame)
        cv2.waitKey(0)

    
