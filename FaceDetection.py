import numpy as np
import time
import cv2
from google.colab.patches import cv2_imshow  #To show images in Colab

def FaceDetectionImage(prototxtpath,modelpath,imagepath,minconfidence):
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt= prototxtpath,caffeModel= modelpath)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(filename= imagepath)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence >  minconfidence:  #args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output image
    cv2_imshow(image) #cv2.imshow("Output",img)  
    cv2.waitKey(0)

print("[INFO] loading Path...")
prototxtpath = "gdrive/My Drive/Colab Notebooks/model/deploy.prototxt.txt"
modelpath = "gdrive/My Drive/Colab Notebooks/model/res10_300x300_ssd_iter_140000.caffemodel"
imagepath = "gdrive/My Drive/Colab Notebooks/Images/fam.png"
minconfidence = 0.5

FaceDetectionImage(prototxtpath,modelpath,imagepath,minconfidence)
