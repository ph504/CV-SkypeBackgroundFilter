# import the opencv library
import cv2
import numpy as np
from matplotlib import pyplot as plt
#import xlsxwriter
#import pandas as pd


# define a video capture object
vid = cv2.VideoCapture(0)

# help(cv2.CascadeClassifier)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def image_process_1(frame):

    imageRgb = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    edges = cv2.Canny(gray, 100, 200)

    blur = cv2.blur(gray, (5, 5))

    blur_edges = cv2.Canny(blur, 50, 200)

    blur_v2 = cv2.blur(gray, (45, 45))

    blur_v3 = cv2.blur(frame, (45, 45))

    dict_of_frames = {"frame": frame,
                    "gray": gray,
                    "edges": edges,
                    "blur": blur,
                    "blur_v2": blur_v2,
                    "blur_v3": blur_v3,
                    "blur_edges" : blur_edges,
                    "imageRgb" : imageRgb}

    return [dict_of_frames, faces]


def main_while():
    while (True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # frame is a numpy.ndarray

        frame_process(frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def frame_process(frame):
    [dict_of_frames, faces] = image_process_1(frame)

    kernel = np.ones((6, 6), np.uint8)
    edge_matrix = dict_of_frames["blur_edges"]
    dilation = cv2.dilate(edge_matrix, kernel, iterations=1)    
    edge_matrix = dilation

    masks = []
    for face in faces:
        masks.append(lineFinder(dict_of_frames["blur_edges"], face))    

    # result = filter_background_v2(dict_of_frames["gray"],dict_of_frames["blur_v2"],masks)
    result = filter_background_v2(dict_of_frames["imageRgb"], dict_of_frames["blur_v3"], masks)
    

    cv2.imshow('frame', dict_of_frames["frame"]) 
    cv2.imshow('gray', dict_of_frames["gray"])
    cv2.imshow('blur', dict_of_frames["blur"])    
    #-------------------------------------------------------
    cv2.imshow('edges', dict_of_frames["edges"])
    cv2.imshow('dilation', dilation)    
    cv2.imshow('blur_edges', dict_of_frames["blur_edges"])
    for mask in masks:
        cv2.imshow('mask', mask)    
    #-------------------------------------------------------
    cv2.imshow('result', result)

def filter_background_v2(image, blurred_image, masks):
    imface = image
    imback = blurred_image
    result = imback
    for mask in masks:
        rows,cols = mask.shape
        for i in range(rows):
            for j in range(cols):
                if(mask[i,j]>0):
                    result[i,j] = imface[i,j]
    return result

def lineFinder(edge_matrix,faceRect):
    (x, y, w, h) = faceRect

    # print(x, y, w, h)
    # print(edge_matrix.shape[0], edge_matrix.shape[1])    


    mask = np.zeros((edge_matrix.shape[0], edge_matrix.shape[1]))
    mask[y:, x:x + w] = 1




    midPoint_x = (int) (x + w/2)
    midPoint_y = (int) (y + h/2)

    # print("alo")

    # left side search
    # first we find a starting point

    threshold_x = 2

    startPoint = (0,0)

    for t in range(0, threshold_x):
        for y_i in range(midPoint_y, edge_matrix.shape[0]):
            if(edge_matrix[y_i][x-t]>0):
                mask[y_i:, x-t] = 1
                startPoint = (y_i,x-t)
                break

    # print(startPoint)
    # print('startPoint')

    currentPoint = startPoint
    endFlag = False

    while (currentPoint[0] < (edge_matrix.shape[0]-1) and currentPoint[1] < (edge_matrix.shape[1]-1) and not (endFlag)):
        (y_c, x_c) = currentPoint
        if (edge_matrix[y_c][x_c - 1] > 0):
            mask[y_c:, x_c] = 1
            currentPoint = (y_c, x_c - 1)
        elif (edge_matrix[y_c + 1][x_c - 1] > 0):
            mask[y_c:, x_c] = 1
            currentPoint = (y_c + 1, x_c - 1)
        elif (edge_matrix[y_c + 1][x_c] > 0):
            currentPoint = (y_c + 1, x_c)
        else:
            endFlag = True

    # right side search

    threshold_x = 2

    startPoint = (0, 0)

    for t in range(0, threshold_x):
        for y_i in range(midPoint_y, edge_matrix.shape[0]):
            if (edge_matrix[y_i][x + w + t] > 0):
                mask[y_i:, x + w + t] = 1
                startPoint = (y_i, x + w + t)
                break

    # print(startPoint)
    # print('startPoint')

    currentPoint = startPoint
    endFlag = False

    while (currentPoint[0] < (edge_matrix.shape[0]-1) and currentPoint[1] < (edge_matrix.shape[1]-1) and not (endFlag)):
        (y_c, x_c) = currentPoint
        if (edge_matrix[y_c][x_c + 1] > 0):
            mask[y_c:, x_c] = 1
            currentPoint = (y_c, x_c + 1)
        elif (edge_matrix[y_c + 1][x_c + 1] > 0):
            mask[y_c:, x_c] = 1
            currentPoint = (y_c + 1, x_c + 1)
        elif (edge_matrix[y_c + 1][x_c] > 0):
            currentPoint = (y_c + 1, x_c)
        else:
            endFlag = True

    # cv2.imshow('mask', mask)
    return mask


main_while()

# frame = cv2.imread('farid1.jpg')
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# blur = cv2.blur(gray, (3, 3))
#
#
# cv2.imshow('frame',frame)
# cv2.waitKey(0)

# plt.subplot(121),plt.imshow(frame),plt.title('frame')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(gray),plt.title('gray')
# plt.xticks([]), plt.yticks([])
# plt.show()


#
# while (True):
#     frame_method(frame)

