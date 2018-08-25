import cv2 as cv
import numpy as np

#######################
# capture the camera #
######################

cap = cv.VideoCapture(0)

#####################
# models of object #
####################

_face = cv.CascadeClassifier('face.xml')
_eyes = cv.CascadeClassifier('eyes.xml')
_smile = cv.CascadeClassifier('smile.xml')

##############
# dimensions #
##############

cw=cap.get(3)
ch=cap.get(4)

print(str(cw)+"-"+str(ch))

#########################
# loop for recognition  #
#########################

while True:
    ###################
    # read the camera #
    ###################
    y, frame = cap.read()

    ##############################################################
    # change the camera color to gray to improve the recognition #
    ##############################################################

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ####################
    # detect the faces #
    ####################
    face = _face.detectMultiScale(gray)
    print("faces: "+str(len(face)))
    #########################################
    # detect faces position, draw and print #
    ########################################
    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        print("face position: "+str(x)+"-"+str(y))

        # I don't have any idea #

        eg = gray[y:y+h,x:x+w]
        ec=frame[y:y+h,x:x+w]

        ###############
        # detect eyes #
        ###############
        eyes = _eyes.detectMultiScale(eg)

        #########################################
        # detect eyes position, draw and print #
        ########################################

        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(ec,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            print("eyes positions: "+str(ex)+"-"+str(ey))
            sg = gray[y:y+h, x:x+w]
            sc=frame[y:y+h,x:x+h]
            ###############
            #detect smile #
            ##############

            smile = _smile.detectMultiScale(sg)

            #########################################
            # detect faces position, draw and print #
            ########################################

            for (sx,sy,sw,sh) in smile:
                cv.rectangle(sc,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
                print("mouse position: "+str(sx)+"-"+str(sy))
    ############################
    # show image on the screen #
    ############################
    cv.imshow("Facial detecion",frame)
    ###########################################
    # if i press the 'q' exit from aplication #
    ###########################################
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
