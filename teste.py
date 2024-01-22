import cv2
print(cv2.__version__)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array


def get_model():
    global model
    model = load_model('model.h5')
    print("Model loaded!")


def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def prediction(img_path):
    new_image = load_image(img_path)
    
    pred = model.predict(new_image)
    
    #print(pred)
    
    labels=np.array(pred)
    labels[labels>=0.6]=1
    labels[labels<0.6]=0
    
    #print(labels)
    final=np.array(labels)
    
    if final[0][0]==1:
        return "Bad"
    else:
        return "Good"
   

get_model()


videopath=0
#@ Let us load the video 

# First we need to create the object of VideoCapture class
cap = cv2.VideoCapture(videopath)
# Check the video initialization is proper or not
status = cap.isOpened()
if status==False:
    print("Error while reading the video..!")
# If everything is fine then go for frame by frame loading
while(True):
    
    # Read the video frame by frame
    retVal, frame = cap.read()    
    cv2.imwrite('frames/frame.jpg', frame)    	             
    print(prediction('frames/frame.jpg'))
    classe = prediction('frames/frame.jpg')
    text = 'Qualidade da placa: '+classe
    
    cv2.putText(frame,text,(0,25), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    os.remove('frames/frame.jpg')
    cv2.putText(frame,'',(0,25), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    # if retVal of above function call is true then show the frame
    if(retVal):
        cv2.imshow("Video",frame)

        # In order to control the video display
        if(cv2.waitKey(25) and 0xFF==27):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()