import \
    face_recognition
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import os
import sys
import logging
import traceback
import ctypes
from typing import Dict
import configparser
import pyttsx3

# Constants
IMAGES_DIRECTORY = None
MAX_ATTEMPTS = 3
TIME_LIMIT = 5
logger = logging.getLogger(__name__)
engine = pyttsx3.init()

def load_images() -> Dict[str, np.array]:
    # Verify if IMAGES_DIRECTORY is set
    if not IMAGES_DIRECTORY:
        raise ValueError("IMAGES_DIRECTORY is not set.")

    # Load jpg/jpeg images from directory
    student_image_paths = list(IMAGES_DIRECTORY.glob("*.jpg")) + list(
        IMAGES_DIRECTORY.glob("*.jpeg")
    )

    # Prepare the dictionary to store the encodings
    student_encodings = {}

    # Iterate over images
    for student_image_path in student_image_paths:
        # Load image and get its face encodings
        image = face_recognition.load_image_file(student_image_path)
        encodings = face_recognition.face_encodings(image)

        # If there is at least one face found, save the first encoding
        if encodings:
            student_encodings[student_image_path.stem] = encodings[0]

    return student_encodings


def match_face_in_frame(frame, student_encodings):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            list(student_encodings.values()), face_encoding, tolerance=0.5
        )

        if True in matches:
            first_match_index = matches.index(True)
            name = list(student_encodings.keys())[first_match_index]
            logger.info(f"Picture identified: Student name is: {name}")
            speak_text(f"Picture identified: Student name is: {name}")
            return True

    return False


def attempt_recognition(video_capture, student_encodings):
    start_time = time.time()

    while time.time() - start_time <= TIME_LIMIT:
        ret, frame = video_capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if match_face_in_frame(frame_rgb, student_encodings):
            return True

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    return False


def recognize_face_from_webcam(student_encodings: Dict[str, np.array]):
    video_capture = cv2.VideoCapture(0)

    for attempt in range(MAX_ATTEMPTS):
        if attempt_recognition(video_capture, student_encodings):
            break

        logger.info(f"Time limit reached for attempt {attempt + 1}")
        speak_text(f"Time limit reached for attempt {attempt + 1}")
    else:
        logger.info(f"No match detected after {MAX_ATTEMPTS} attempts.")
        speak_text(f"No match detected after {MAX_ATTEMPTS} attempts.")

    video_capture.release()
    cv2.destroyAllWindows()


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Read the configuration file
    config = configparser.ConfigParser()
    if args.config_file:
        config.read(args.config_file)
    elif getattr(sys, 'frozen', False):
        config.read('config.ini')
    else:
        raise ValueError("Configuration file path is not specified.")

    global IMAGES_DIRECTORY
    IMAGES_DIRECTORY = Path(config.get('Settings', 'IMAGES_DIRECTORY'))

    # Verify if IMAGES_DIRECTORY is set
    if not IMAGES_DIRECTORY:
        raise ValueError("IMAGES_DIRECTORY is not set.")

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Create log file handler
    log_file = os.path.join(IMAGES_DIRECTORY, "facerecognition.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    file_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Add a stream handler to also output logs to the console (only if not added before)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set the desired log level for console output
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.info(f"IMAGES_DIRECTORY: {IMAGES_DIRECTORY}")
    speak_text(f"IMAGES_DIRECTORY: {IMAGES_DIRECTORY}")

    logger.info("Loading student images...")
    speak_text("Loading student images...")
    student_encodings = load_images()

    logger.info("Starting face recognition...")
    speak_text("Starting face recognition...")

    try:
        recognize_face_from_webcam(student_encodings)
        logger.info("Face recognition completed successfully.")
        speak_text("Face recognition completed successfully.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n\n{traceback.format_exc()}"
        logger.error(error_message)
        ctypes.windll.user32.MessageBoxW(0, error_message, "Error", 0x10)
        speak_text("An error occurred during face recognition.")

    logger.removeHandler(file_handler)  # Remove the console handler after completion


if __name__ == "__main__":
    main()
