import cv2

class merge:
    
    def __init__(self):
        pass

    def merg(self,img):

        layer=img.copy()
        pyramids=[layer]
        for i in range(6):
            layer=cv2.pyrDown(layer)
            pyramids.append(layer)
        layer=pyramids[5]
        lap=[layer]
        for i in range(5,0,-1):
            size = (pyramids[i - 1].shape[1], pyramids[i - 1].shape[0])
            lap.append(cv2.subtract(pyramids[i-1],cv2.pyrUp(pyramids[i],dstsize=size)))
        rec=lap[0]
        for i in range(1,6):
            rec=cv2.add(cv2.pyrUp(rec,dstsize=(lap[i].shape[1],lap[i].shape[0])),lap[i])
        return rec

