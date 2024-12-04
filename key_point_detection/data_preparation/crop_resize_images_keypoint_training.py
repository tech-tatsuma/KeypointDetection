from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use( 'tkagg' )

def detection_gauge_face(img, img_path, model_path, box_index=-1):
    
    model = YOLO(model_path) 

    results = model(img) 

    boxes = results[0].boxes
    
    if box_index>=0:
        m_box = boxes[box_index]
    else:     
        m_box = boxes[0]
    return m_box.xyxy[0].int(), boxes


def crop_image(img, box):
    img = np.copy(img)
    cropped_img = img[box[1]:box[3],
                      box[0]:box[2], :]  # image has format [y, x, rgb]
    
    height = int(box[3]-box[1])
    width = int(box[2]-box[0])
    
    print(f"Height is {height}, Width is {width}")
    # want to preserve aspect ratio but make image square, so do padding
    if height > width:
        delta = height-width
        left, right = delta//2, delta - (delta//2)
        top = bottom = 0
    else:
        delta = width-height
        top, bottom = delta//2, delta - (delta//2)
        left = right = 0
            
    pad_color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return new_img


def process_image(img_path, model_path, box_index=-1):
    image = cv2.imread(img_path)
    print(f"Image path is {img_path}")
    print(f"Image shape is {image.shape}")

    # Gauge detection
    box, boxes = detection_gauge_face(image, img_path, model_path, box_index= box_index)

    # crop image to only gauge face
    cropped_img = crop_image(image, box)
    
    resolution = (448,448)
    resized_img = cv2.resize(cropped_img, resolution, interpolation = cv2.INTER_LINEAR)

    return resized_img, boxes, image

def get_files_from_folder(folder):
    filenames = []
    for filename in os.listdir(folder):
        filenames.append(filename)
    return filenames

def crop_and_save_img(filename, src_dir, dest_dir, model_path, box_index=-1):
    img_path = src_dir + filename

    cropped_img, boxes, image = process_image(img_path, model_path, box_index)
    
    new_file_path = os.path.join(dest_dir, 'cropped_'+ filename)
    cv2.imwrite(new_file_path, cropped_img)
    
def plot_bounding_box_img(img, boxes):
    """
    plot detected bounding boxes. boxes is the result of the yolov8 detection
    :param img: image to draw bounding boxes on
    :param boxes: list of bounding boxes
    """
    for box in boxes:
        bbox = box.xyxy[0].int()
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))

        color_face = (0, 255, 0)
        color_needle = (255, 0, 0)
        if box.cls == 0:
            color = color_face
        else:
            color = color_needle

        img = cv2.rectangle(img,
                            start_point,
                            end_point,
                            color=color,
                            thickness=5)

    plt.figure()
    plt.imshow(img)
    plt.show()

if __name__=="__main__":
    image_directory = '/home/furuya/analog-gauge-reader/analog_gauge_reader/testdata/'
    new_image_directory = '/home/furuya/analog-gauge-reader/analog_gauge_reader/testdata/data/'
    model_path = "/home/furuya/analog-gauge-reader/analog_gauge_reader/gauge_detection/runs/detect/train/weights/best.pt"

    test_file_names = get_files_from_folder(image_directory)

    os.makedirs(new_image_directory, exist_ok=True)

    for filename in test_file_names:
        crop_and_save_img(filename, image_directory, new_image_directory, model_path)

    i=1
    for filename in os.listdir(new_image_directory):
        os.rename(new_image_directory + filename, new_image_directory + str(i) + "_" + filename)
        i+=1

    # resize
    resolution = (448,448)
    for filename in os.listdir(new_image_directory):
        img_path = os.path.join(new_image_directory, filename)
        image = cv2.imread(img_path)
        resized_img = cv2.resize(image, resolution, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(img_path, resized_img)