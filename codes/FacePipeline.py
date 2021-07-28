from deepface import DeepFace
import cv2
import numpy as np
import os
import csv


'PIPELINE TO AUTOMATICALLY DETECT AND IDENTIFY FACES USING DEEPFACE MODEL, CLUSTER FACES AND SAVE INTO CSV '


class FaceDetector:

    def __init__(self, protxt='deploy.prototxt.txt',
                 model='res10_300x300_ssd_iter_140000.caffemodel', face_id=0):

        self.net = cv2.dnn.readNetFromCaffe(protxt, model)
        self.face_id = face_id

    def detect(self, frame):
        (height, width) = frame.shape[:2]
        self.blob = cv2.dnn.blobFromImage(frame, 1, (300, 300))
        self.net.setInput(self.blob)
        faces = self.net.forward()
        box_list = []

        for i in range(0, faces.shape[2]):
            confidence = faces[0,0,i,2]

            if confidence > 0.25:   # Change Parameter
                box = faces[0,0,i, 3:7] * np.array([width, height, width, height])
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])

                box_list.append((x, y, w, h, confidence))

        return box_list

    def identify(self, frame, box_list, image_name, folder_face_db, row_list, episode_id, image_path):

        face_folder = os.path.join(os.getcwd(), 'face_db_test', folder_face_db)
        scale_factor = 1.2

        temp_folder = os.path.join(os.getcwd(), 'temp')

        for person in box_list:
            faces_list = [os.path.join(face_folder, file) for file in os.listdir(face_folder) if file.__contains__('.jpg')]
            x = int(person[0])
            y = int(person[1])
            w = int(person[2])
            h = int(person[3])

            face_bb = (x, y, w, h)

            roi_color = frame[int(y / scale_factor):int((y + h) * scale_factor), int(x / scale_factor):int((x + w)* scale_factor)]

            if roi_color.shape[0] and roi_color.shape[1] and roi_color.shape[2] != 0:
                pass

            elif roi_color.shape[0] or roi_color.shape[1] or roi_color.shape[2] == 0:
                print("Problem BB: ", face_bb)
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                roi_color = frame[int(y):int((y + h)), int(x):int((x + w))]
                if roi_color.shape[0] and roi_color.shape[1] and roi_color.shape[2] != 0:
                    print("Problem Solved!")
                    pass
                else:
                    print("NO ROI Found", image_name, roi_color.shape)
                    continue

            if len(faces_list) == 0:
                crop_path = os.path.join(face_folder,
                                         str(image_name[:-4]) + '_' + 'face' + str(self.face_id) + '.jpg')
                cv2.imwrite(crop_path, roi_color)
                new = crop_path.split('\\', 9)[8]
                print("New Face: ", new, face_bb)
                intermediate_list = [image_path, 1,
                                     "{" + f"\"episode\":{episode_id}" + "}", 1, 0,
                                     "{" + f"\"name\":\"rect\",\"x\":{x},\"y\":{y},\"width\":{w},\"height\":{h}" + "}",
                                     "{" + f"\"person_id\":{self.face_id}" + "}"]
                row_list.append(intermediate_list)
                self.face_id += 1
                # print("First Image", roi_color.shape, crop_path[-12:])

            else:
                temp_path = os.path.join(temp_folder, str(image_name))
                # print("Subsequent Image",roi_color.shape, temp_path[-12:])
                if roi_color.shape[0] and roi_color.shape[1] and roi_color.shape[2] != 0:
                    cv2.imwrite(temp_path, roi_color)
                else:
                    print("Error!", temp_path)
                    break

                for face in faces_list:
                    result = DeepFace.verify(temp_path, face, enforce_detection=False)
                    if result['verified'] is True:
                        old = face.split('\\', 9)[8].split("_")[-1][:-4][4:]
                        new = temp_path.split('\\', 8)[7]
                        print("Match Found!", new, "Bounding Box: ", face_bb, " --> ", old)  # new --> old
                        intermediate_list = [image_path, 1,
                                             "{" + f"\"episode\":{episode_id}" + "}", 1, 0,
                                             "{" + f"\"name\":\"rect\",\"x\":{x},\"y\":{y},\"width\":{w},\"height\":{h}" + "}",
                                             "{" + f"\"person_id\":{old}" + "}"]
                        row_list.append(intermediate_list)
                        os.remove(temp_path)
                        break

                    elif result['verified'] is False and face == faces_list[len(faces_list) - 1]:
                        crop_path = os.path.join(face_folder,
                                                 str(image_name[:-4]) + '_' + 'face' + str(self.face_id) + '.jpg')
                        cv2.imwrite(crop_path, roi_color)
                        new = crop_path.split('\\', 9)[8]
                        print("New Face: ", new, face_bb)
                        intermediate_list = [image_path, 1,
                                             "{" + f"\"episode\":{episode_id}" + "}", 1, 0,
                                             "{" + f"\"name\":\"rect\",\"x\":{x},\"y\":{y},\"width\":{w},\"height\":{h}" + "}",
                                             "{" + f"\"person_id\":{self.face_id}" + "}"]
                        row_list.append(intermediate_list)
                        self.face_id += 1
                        os.remove(temp_path)

                    else:
                        pass


    def write_to_csv(self, field_list, row_list, csv_name='egosocial_dataset_test.csv'):
        file_exists = os.path.isfile(os.path.join(os.getcwd(), 'face_db_test', csv_name))
        file_name = os.path.join(os.getcwd(), 'face_db_test', csv_name)

        with open(file_name, 'a', newline='') as f:
            write = csv.writer(f)

            if not file_exists:
                write.writerow(field_list)

            write.writerows(row_list)


face_detector = FaceDetector()
#image_root = 'D://EgoSocialStyle/trainval/imageSets/'
image_root = 'D://EgoSocialStyle/test/imageSets/'    # CHANGE IMAGE ROOT HERE
image_folder_list = [folder for folder in os.listdir(image_root)]

for image_folder in image_folder_list:  # Folder 1, 2, 3, ...
    print("Folder ", image_folder)
    images_list = [file for file in os.listdir(os.path.join(image_root, image_folder, 'data')) if file.__contains__('.jpg')]
    # add 'data' for test folder only
    os.makedirs(os.path.join(os.getcwd(), 'face_db_test', image_folder), exist_ok=True)

    field_list = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
    row_list = []

    for img in images_list:
        frame = cv2.imread(os.path.join(image_root, image_folder, 'data', img))  # add 'data' for test folder only
        box_list = face_detector.detect(frame)
        face_detector.identify(frame, box_list, img, image_folder, row_list=row_list, episode_id=image_folder,
                               image_path=os.path.join(image_root, image_folder, img))

    face_detector.write_to_csv(field_list, row_list)
