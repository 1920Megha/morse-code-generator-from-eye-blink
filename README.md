# morse-code-generator-from-eye-blink
Morse code is a a system of communication developed by Samuel Morse in which the letters of the alphabet are coded as a combination of dots and dashes so that messages can either be sent using light, sound or wireless or by EYE BLINK. Morse Code encodes the ISO basic alphabet, numerals and a small set of punctuation and procedural signals (prosigns) as standardized sequences of short and long signals called "dots" and "dashes", or "dits" and "dahs", as in amateur radio practice.

# Importance behind Morse code
In this type of communication, people will communicate using eye blinks according to their situation or convenience. Every eye blink will have a certain meaning and according to that, people will communicate. This was the quickest long-distance method of communication at the point of its invention why because a single gesture in more code will convey a lot of information. Morse code played an important role in information passing during the Second World War Since it increased the speed of communication and it cannot be encoded by all people. 

# why morse code
Inspired from Google's Experiment about how they used morse code to help differently abled people to communicate efficiently. I decided to implement morse code translator using computer vision which is cheaper and efficient option as the code generated is also stored in form of audio and text formates for convinence of people and can be used by differently abled people to communicate.

# Working
This project translates morse code in plain english. We used webcam to read blinking of the eyes as dots and dashes which then with the use of a dictionary converts morse to english. I have used jetson nano kit of nvidia for this project

File morse_converter.py contains the python dictionary. For reference I have also included the Dictonary to learn Morse better.
- Short Blink : Dot ' . '
- Long Blink : Dash ' - '
- Too Long Blink:Next Word

![](https://github.com/1920Megha/morse-code-generator-from-eye-blink/blob/main/Eye%20blink.gif)
# Required libraries
1)OpenCv
2)imutils
3)dlib
4)Scipy

# How to run
Install libraries specified.Then clone this repository or Download and run following:

To run in command lne/termnal
-python eye_blink_Morse.py --shape-predictor shape_predictor_68_face_landmarks.dat
