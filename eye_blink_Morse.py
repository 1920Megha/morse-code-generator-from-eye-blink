#command to run
# python3 eye_blink_Morse.py --shape-predictor shape_predictor_68_face_landmarks.dat

# necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from gtts import gTTS
import os
def MorseToText(s):

    mDict={
        '' : 'Check',
    '.-': 'a',
    '-...': 'b',
    '-.-.': 'c',
    '-..': 'd',
    '.': 'e',
    '..-.': 'f',
    '--.': 'g',
    '....': 'h',
    '..': 'i',
    '.---': 'j',
    '-.-': 'k',
    '.-..': 'l',
    '--': 'm',
    '-.': 'n',
    '---': 'o',
    '.--.': 'p',
    '--.-': 'q',
    '.-.': 'r',
    '...': 's',
    '-': 't',
    '..-': 'u',
    '...-': 'v',
    '.--': 'w',
    '-..-': 'x',
    '-.--': 'y',
    '--..': 'z',
    '.-.-': ' ',
    '-----':'0',
    '.----':'1',
    '..---':'2',
    '...--':'3',
    '....-':'4',
    '.....':'5',
    '-....':'6',
    '--...':'7',
    '---..':'8',
    '----.':'9',
    '..--..':'?',
    '-.-.--':'!',
    '.-.-.-':'.'
    }

    if mDict.get(s)!='Check':
    	return str(mDict.get(s))
		


def aspect_ratio_of_eye(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# frames the eye must be below the threshold
EYE_THRESH = 0.27
EYE_CONSEC_FRAMES = 3
EYE_CONSEC_FRAMES2=6
EYE_CONSEC_FRAMES3=10

COUNTER = 0
TOTAL=[]


string=""   #Morse Message

language = 'en'


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print(".............. facial landmark predictor is getting load wait few seconds.........")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("\n......... starting video stream thread wait few seconds..........")


def gstreamer_pipeline(
	capture_width=1280,
	capture_height=720,
	display_width=1280,
	display_height=720,
	framerate=30,
	flip_method=0,
):
	return (
		"nvarguscamerasrc ! "
		"video/x-raw(memory:NVMM), "
		"width=(int)%d, height=(int)%d, "
		"format=(string)NV12, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink"
	      % (
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
	  	)
  )


key=cv2.waitKey(1)

#for external cameras 
#vs =cv2.VideoCapture('http://172.168.8.44:8080/video')
vs=cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) #for jetson camera

#for laptop inbuilt camera
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
#vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
#time.sleep(1)

# loop over frames from the video stream
while True:

	ret,frame = vs.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0) # detect faces in the grayscale frame

	# loop over the face detections
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = aspect_ratio_of_eye(leftEye)
		rightEAR = aspect_ratio_of_eye(rightEye)

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_THRESH:
			COUNTER += 1

		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_CONSEC_FRAMES and COUNTER<=EYE_CONSEC_FRAMES2:
				TOTAL.append(".")
			elif COUNTER >=EYE_CONSEC_FRAMES2 and COUNTER<=EYE_CONSEC_FRAMES3:
				TOTAL.append("-")
			elif COUNTER>=EYE_CONSEC_FRAMES3:
				s=str(MorseToText(''.join(TOTAL)))
				if s=="None":
					TOTAL=[]
				else:
					string+=s
					TOTAL=[]

				
			COUNTER = 0

		#output on screen
		cv2.putText(frame, "Morse_Code: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		#cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame,"Messsage: {}".format(string), (10,50), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame,"", (10,300), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		cv2.putText(frame,"short blink= . | long blink = - |too long blink= end", (10,320), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		
		cv2.putText(frame,"press 'q' to quit", (20,70), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# i`q`= break 
	if key == ord("q"):
		break


# saving in text or audio
save_path='audio/'
myobj = gTTS(text=string, lang=language, slow=False)
aud_file_name = str(save_path)+str(string) + ".mp3"
txt_file_name=str(save_path)+str(string) + ".txt"
myobj.save(aud_file_name)
file1 = open(txt_file_name, "w")
file1.write(string)
file1.close()
#os.system(aud_file_name)  #playing it
print("\n******** audio and text saved in audio folder*********")
# do a bit of cleanup
cv2.destroyAllWindows()

