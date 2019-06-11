from django.shortcuts import render
from django.http import Http404
from django.conf import settings

#project imports start
import cv2
import sys
import os
import re
import imutils
import shutil
import numpy as np
import pandas as pd
import pytesseract
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
#project import end

def home(request):
    return render(request, 'index.html')

def select_video(request):
    return render(request, 'selectVideo.html')

def recognition(request):
    if request.method == 'POST' and request.FILES:

        #Start project process
        frame_dir = os.path.join(settings.BASE_DIR, 'media', 'frames')
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        else:
            shutil.rmtree(os.path.join('media', 'frames'))
            os.mkdir(frame_dir)

        file = request.FILES.get('video_file')
        fname = file.temporary_file_path()

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        def preprocess_frame(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.bilateralFilter(gray, 11, 17, 17)
            canny_edge = cv2.Canny(blur, 170, 200)

            (new, contours, _) = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
            plate_count = None
            status = False


            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    plate_count = approx
                    break

            if isinstance(plate_count,np.ndarray) :
                kernel = np.zeros(gray.shape,np.uint8)
                num_plt_image = cv2.drawContours(kernel,[plate_count],0,255,-1)
                num_plt_image = cv2.bitwise_and(frame,frame,mask=kernel)
                status = True
            else:
                num_plt_image = frame

            th, binary = cv2.threshold( num_plt_image, 127,255,cv2.THRESH_BINARY );
        # cv2.imwrite('binary.png',binary)
            return binary, status

        def convert2Frame(outDir, fname):
            vidcapture = cv2.VideoCapture(fname)
            success,image = vidcapture.read()
            count = 0
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            while success:
                image, stat = preprocess_frame(imutils.rotate(image,-90))
                if stat == True:
                    cv2.imwrite(outDir+"\\%d.png" % count, image)
                success,image = vidcapture.read()
                count += 1
            return count

        frame_count = convert2Frame(frame_dir, fname)
        model = cv2.dnn.readNet(os.path.join(settings.BASE_DIR, 'core', 'files', 'deep_model.pb'))

        def detect_text_from_image(img_path, model=model):
            image = cv2.imread(img_path)

            orig_image = image.copy()
            (H, W) = image.shape[:2]

            (newW, newH) = (320, 320)
            rW = W / float(newW)
            rH = H / float(newH)

            image = cv2.resize(image, (320, 320))
            (H, W) = image.shape[:2]

            layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"
            ]
            blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
            model.setInput(blob)
            (scores, geometry) = model.forward(layerNames)

            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []

            for y in range(0, numRows):
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                for x in range(0, numCols):
                    if scoresData[x] < 0.5:
                        continue

                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]

                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            boxes = non_max_suppression(np.array(rects), probs=confidences)

            count = 0
            for (startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                plate = orig_image[startY:endY, startX:endX]
                cv2.imwrite(str(count)+'.png',plate)
                count += 1
            return plate

        def get_text_from_frame(imgpath):
            image = cv2.imread(imgpath)
            text = pytesseract.image_to_string(image, config=('-l eng --oem 1 --psm 3'))
            return imgpath, text

        data, dict_count = [], {}
        for frame in range(frame_count):
            filename=str(frame)+".png"
            fname = os.path.join(settings.BASE_DIR, 'media', 'frames', filename)
            if os.path.isfile(fname):
                img_path, text = get_text_from_frame(fname)
                if len(text) > 5:
                    text = ''.join(e for e in text.strip() if e.isalnum())
                    if text in dict_count:
                        dict_count[text] += 1
                    else:
                        dict_count[text] = 1
                    data.append((filename, text))
        result = None
        max_count = 0
        re_card = r"^[a-zA-Z][a-zA-Z]\d\d[a-zA-Z][a-zA-Z]\d\d\d\d$"
        for res in dict_count:
            if not result:
                result = res
            if re.match(re_card, res) and dict_count[res] > max_count:
                max_count = dict_count[res]
                result = res
        return render(request, 'recognition.html', {'data': data, 'result': result})
    raise Http404
