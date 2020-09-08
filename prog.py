import cv2
import numpy as np


# Reads in the weights and the configuration files
net = cv2.dnn.readNet('C:/Users/alecy/OneDrive/Desktop/randpython/yolov3.weights', 'C:/Users/alecy/OneDrive/Desktop/randpython/yolov3.cfg.txt')

# get list of classes names
classes = []
with open('C:/Users/alecy/OneDrive/Desktop/randpython/names.txt', 'r') as f:
    classes = f.read().splitlines()

# Read in an image to process
img = cv2.imread('C:/Users/alecy/OneDrive/Desktop/randpython/images/unimelb.jpg')
height, width, _ = img.shape

print(height, width)

# Create a blob object, that is scaled correctly as input for the model
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)

# To see what the input image(s) look like.
# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob)

# Set the new input value for the network
net.setInput(blob)

# Run the forward pass to compute the output layer from the neural net
# Output layer should contain the same number as len(blob)??
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Get the results, predictted classes, Confidences
boxes = []
confidences = []
class_ids = []

# Extract the relevant data from the layerOutput
# For each detections list
for output in layerOutputs:
    # For each list of results:
    for detection in output:
        # A result has 4 coordinates for a box, a confidence for the box
        # and the probabilities of each label being the image in the box
        scores = detection[5:] # Get the probs of each label
        class_id = np.argmax(scores) # find the most probable class label
        confidence = scores[class_id] # Find the confidence of the most confidence class
        if confidence > 0.4:
            # The first 4 elements in detection give the coordinates of the box outline
            # scale these numbers to the original image

            # Coords and size of image box
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # We need this for open CV's use. needs the position of the upper left corner
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# We detect this many objects in the image
print(len(boxes))

# Filter boxes by scores, and suppressing non-maximum probabilities. numbers: confidence, threshhold
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

# Shows which boxes "made the cut". len(boxes) - length of this = number of redundant boxes
print(indexes.flatten())

# Font and color of the boxes. RGB color tuples
font = cv2.FONT_HERSHEY_PLAIN
font_size = 1
font_thick = 1
colors = np.random.uniform(0, 255, size = (len(boxes), 3))

# If there are image boxes, then draw out the boxes and label them
if len(indexes.flatten()) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2) # last param is thickness of box
        cv2.putText(img, label + " " + confidence, (x, y - 5), font, font_size, (255, 255 ,255), font_thick)

# Can resize the image if its too small or big
img = cv2.resize(img, (1800, 1000))

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(layerOutputs)
