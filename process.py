import preprocess 
import cv2
import numpy as np
from tensorflow.keras import models

#get model
model = models.load_model('./models/modelRD.h5')
model.load_weights('./models/weight_of_modelRD.h5')
#get image
img = cv2.imread('./img/img1.jpg')
imgMaxContrast = preprocess.maxContrast(img)
data = preprocess.getData(imgMaxContrast)
#predict 
y_predict = model.predict(data)
result = np.argmax(y_predict, axis = 1)
final_string = ''
for num in result:
    final_string += str(num)
#show image
img = cv2.putText(img, final_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
cv2.imshow('a',img)
cv2.waitKey(0)