"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import base64
import os
from io import BytesIO
# Python default libraries
from pathlib import Path

# processing images
import cv2
# Face Detector
import face_detection
import numpy
import numpy as np
# masked or not masked classification
import tensorflow as tf
import typer
from PIL import Image, ImageDraw, ImageFont

# visualising the data

# Max enabled image width is set as 300. If greater we will resize the input images


# Might affect quality of the prediction, it will be scaled for faster processing
INTERMEDIATE_WIDTH = 200
MASKED_COLOR = (137, 177, 0)
NOT_MASKED_COLOR = (189, 90, 54)

ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))
FONT_TTF_LOC = str(Path(ROOT_DIR) / 'face-mask-detection' / 'data' / 'fonts' / 'Arvo-Regular.ttf')

face_detector = face_detection.build_detector('RetinaNetMobileNetV1',
                                              confidence_threshold=.5,
                                              nms_iou_threshold=.3)


def resize_image(img, basewidth):
    # Resize image by keeping the aspect ratio if image witdth is greater than BASEWIDTH
    # if img.size[0] > basewidth:
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def detect_face(resized_img):
    open_cv_image = np.array(resized_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert BGR to RGB
    return face_detector.detect(open_cv_image)


classifier_dir = Path(ROOT_DIR) / 'face-mask-detection' / 'data' / 'classifier_model_weights'
classifier = tf.keras.models.load_model(classifier_dir / 'best.h5')


def classify_faces(img_raw, face_coords):
    classification_scores = []
    # Iterate over detected face coordinates to find
    for coords in face_coords:
        x1, y1, x2, y2, _ = coords
        cropped_face = img_raw.crop((x1, y1, x2, y2))
        img = np.float32(cropped_face)
        img = cv2.resize(img, (112, 112))
        preprocessed_img = tf.keras.applications.mobilenet.preprocess_input(img)
        preprocessed_img = preprocessed_img[np.newaxis, ...]
        pred = classifier.predict_on_batch(preprocessed_img)[0][0]
        classification_scores.append(pred)
    return classification_scores


def annotate_image(img, face_coords, classified_face_scores, classification_labels):
    pil_draw = ImageDraw.Draw(img)
    for idx, coords in enumerate(face_coords):
        x1, y1, x2, y2, _ = coords
        label = classification_labels[idx]
        color = MASKED_COLOR if label == 'masked' else NOT_MASKED_COLOR
        display_str = "{}: {:.2f}".format(label, classified_face_scores[idx])

        # Draw rectangle for detected face
        pil_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label text box
        # portion of image width you want text width to be
        img_fraction = 0.2
        font_size = 5  # starting font size

        font = ImageFont.truetype(FONT_TTF_LOC, font_size)
        image_size = img.size[0]

        while font.getsize(display_str)[0] < img_fraction * image_size:
            # iterate until the text size is just larger than the criteria
            font_size += 1
            font = ImageFont.truetype(FONT_TTF_LOC, font_size)

        # Find coordinates of bounding text box
        w, h = font.getsize(display_str)
        pil_draw.rectangle([x1, y1, x1 + w, y1 + h], fill=color)
        pil_draw.text((x1, y1), display_str, font=font)
    return img


def convert_pil_to_base64(annotated_image, image_type):
    buffered = BytesIO()
    if image_type == 'jpg':
        annotated_image.save(buffered, format='jpeg')
    elif image_type == 'png':
        annotated_image.save(buffered, format='png')
    else:
        annotated_image.save(buffered, format=image_type)
    annotated_image_base64 = base64.b64encode(buffered.getvalue())
    return annotated_image_base64.decode('utf-8')


def predict_masked_faces(image, intermediate_width):
    # Resize image for performance
    resized_img = resize_image(image, intermediate_width)

    # Detect face coordinates from the raw image
    face_coords = detect_face(resized_img)

    # Classify detected faces whether they have a mask or not
    classified_face_scores = classify_faces(resized_img, face_coords)

    # Find labels
    classification_labels = np.where(np.array(classified_face_scores) > 0.5, 'masked', 'not masked').tolist()

    # Convert score to string type to make it serializable
    classified_face_scores = [float(score) for score in classified_face_scores]

    return {
        'detected_face_coordinates': face_coords,
        'detected_mask_scores': classified_face_scores,
        'detected_face_labels': classification_labels
    }


def show_webcam(mirror, intermediate_width):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        # Convert to PIL
        img_raw = Image.fromarray(img, 'RGB')

        response = predict_masked_faces(img_raw, intermediate_width)
        # annotated_image = response['annotated_image']
        wpercent = (float(img_raw.size[0]) / intermediate_width)
        annotated_image = annotate_image(
            img_raw,
            response["detected_face_coordinates"] * wpercent,
            response["detected_mask_scores"],
            response["detected_face_labels"]
        )
        array_image = numpy.asarray(annotated_image)
        cv2.imshow('Face Mask Detection', array_image)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main(mirror: bool = False, intermediate_width: int = 200):
    show_webcam(mirror=mirror, intermediate_width=intermediate_width)


if __name__ == '__main__':
    typer.run(main)
