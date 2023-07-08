import face_recognition
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# Load the gender model
gender_model_path = r"C:\Users\HP\Downloads\gender_deploy.prototxt"
gender_model_weights_path = r"C:\Users\HP\Downloads\gender_net.caffemodel"
gender_net = cv.dnn.readNetFromCaffe(gender_model_path, gender_model_weights_path)

# Open the file dialog to select the target image
Tk().withdraw()
load_image = askopenfilename()

# Load the target image and encode the face
target_image = face_recognition.load_image_file(load_image)
target_encoding = face_recognition.face_encodings(target_image)[0]

print(target_encoding)

# Define the gender labels
gender_list = ['Male', 'Female']


def encode_faces(folder):
    # Encode the faces in the provided folder
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_image = face_recognition.load_image_file(os.path.join(folder, filename))
        known_encoding = face_recognition.face_encodings(known_image)[0]
        list_people_encoding.append((known_encoding, filename.split(".")[0]))

    return list_people_encoding


def find_target_face():
    # Find face locations and encodings in the target image
    face_locations = face_recognition.face_locations(target_image)
    face_encodings = face_recognition.face_encodings(target_image, face_locations)
    matched_persons = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matched_names = []

        for person in encode_faces('folder_photo'):
            encoded_face = person[0]
            filename = person[1]

            # Compare face encodings to check for a match
            is_target_face = face_recognition.compare_faces([encoded_face], face_encoding, tolerance=0.55)
            print(f'{is_target_face}{filename}')

            if any(is_target_face):
                matched_names.append(filename)

        if matched_names:
            label = ', '.join(matched_names)
        else:
            label = "Unknown Person"

        # Perform gender prediction on the face
        top, right, bottom, left = face_location
        face_image = target_image[top:bottom, left:right]

        blob = cv.dnn.blobFromImage(face_image, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)
        gender_output = gender_net.forward()

        gender_label = gender_list[gender_output[0].argmax()]

        create_frame(face_location, label, gender_label)

    cv.imshow("Target Image", target_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def create_frame(face_location, label, gender_label):
    # Create a frame around the face and print the label and gender
    top, right, bottom, left = face_location

    # Draw a rectangle around the face
    cv.rectangle(target_image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Define font properties
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Print label and gender on the frame
    cv.putText(target_image, label, (left, top - 20), font, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)
    cv.putText(target_image, gender_label, (left, bottom + 20), font, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)


# Call the find_target_face function
find_target_face()



# import face_recognition
# import cv2 as cv
# import numpy as np
# import os
#
# # Load the gender model
# gender_model_path = r"C:\Users\HP\Downloads\gender_deploy.prototxt"
# gender_model_weights_path = r"C:\Users\HP\Downloads\gender_net.caffemodel"
# gender_net = cv.dnn.readNetFromCaffe(gender_model_path, gender_model_weights_path)
#
# # Define the gender labels
# gender_list = ['Male', 'Female']
#
# # Load the known faces and their encodings
# known_faces = []
# known_names = []
#
# for filename in os.listdir('folder_photo'):
#     image = face_recognition.load_image_file(os.path.join('folder_photo', filename))
#     face_encoding = face_recognition.face_encodings(image)[0]
#     known_faces.append(face_encoding)
#     known_names.append(filename.split(".")[0])
#
# # Initialize the video capture
# cap = cv.VideoCapture(0)
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Convert the frame to RGB
#     rgb_frame = frame[:, :, ::-1]
#
#     # Find all face locations and encodings in the current frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = []
#
#     # Iterate through each face in the current frame
#     for face_location in face_locations:
#         # Compare the face encoding with the known faces
#         face_encoding = face_recognition.face_encodings(rgb_frame, [face_location], num_jitters=10)[0]
#         face_encodings.append(face_encoding)
#
#     # Iterate through each face encoding in the current frame
#     for face_encoding, face_location in zip(face_encodings, face_locations):
#         # Compare the face encoding with the known faces
#         matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.55)
#         name = "Unknown"
#
#         # Find the best match
#         if True in matches:
#             match_index = matches.index(True)
#             name = known_names[match_index]
#
#         # Extract the face coordinates
#         top, right, bottom, left = face_location
#
#         # Perform gender prediction on the face
#         face_image = frame[top:bottom, left:right]
#         blob = cv.dnn.blobFromImage(face_image, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#         gender_net.setInput(blob)
#         gender_output = gender_net.forward()
#         gender_label = gender_list[gender_output[0].argmax()]
#
#         # Draw a rectangle around the face and display the name and gender
#         cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv.putText(frame, name, (left, top - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv.putText(frame, gender_label, (left, bottom + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv.imshow('Face Recognition', frame)
#
#     # Break the loop when 'q' is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture and close all windows
# cap.release()
# cv.destroyAllWindows()
