import yolov5
import imageio
import os
import cv2


img_sources = './images'

# load model

model = yolov5.load('keremberke/yolov5m-license-plate')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image



def process_image(img_file, cropped_dest = './cropped'):

  # perform inference
  #results = model(img_file, size=640)

  # inference with test time augmentation
  results = model(img_file, augment=True)

  # parse results
  predictions = results.pred[0]
  boxes = predictions[:, :4] # x1, y1, x2, y2
  scores = predictions[:, 4]
  categories = predictions[:, 5]

  if len(boxes) > 0:
    x1 = round(float(boxes[0][0]))
    y1 = round(float(boxes[0][1]))
    x2 = round(float(boxes[0][2]))
    y2 = round(float(boxes[0][3]))


    # crop the box and save as an image
    image = cv2.imread(img_file)
    cropped = image[y1: y2, x1:x2, :]
    dest_path = os.path.join(cropped_dest, f"cropped_{img_file.split('/')[-1]}")
    imageio.imwrite(dest_path, cropped)


    ########### save annotations ################
    with open('annotations.csv', 'a') as f:
      line = f"{img_file},{x1},{x2},{y1},{y2},{float(scores[0])},{dest_path}\n"
      f.writelines([line])

    
  else:
    with open('annotations.csv', 'a') as f:
      line = f"{img_file},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR\n"
      f.writelines([line])



import os
from tqdm import tqdm

image_files = os.listdir(img_sources)
for image_file in tqdm(image_files):
  process_image(f"./images/{image_file}")