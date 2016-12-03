# Importing some useful libarries
import numpy as np
import cv2
import math

# Lane co-ordinates
lane1X1, lane1Y1, lane1X2, lane1Y2 = 0,0,0,0
lane2X1, lane2Y1, lane2X2, lane2Y2 = 0,0,0,0

def LaneDetect(image):
    global lane1X1, lane1Y1, lane1X2, lane1Y2
    global lane2X1, lane2Y1, lane2X2, lane2Y2

    # TODO: put image pipeline here,
    # Return the final output (image with lines are drawn on lanes)

    # Get image height and width
    h, w = image.shape[0], image.shape[1]
    # Find ROI part
    offsetX = h/2 + 50
    ROI = image[offsetX:,:]
    # converting to gray scale
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    #print plt.imshow(gray, cmap = 'gray')

    # remove noise
    img = cv2.GaussianBlur(gray,(5,5),0)
    #print plt.imshow(img, cmap = 'gray')

    edges = cv2.Canny(img,150,150) #apertureSize = 3)
    #print plt.imshow(edges, cmap = 'gray')
    # Display the resulting frame
    #cv2.imshow('edges_frame',edges)

    '''
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,3,np.pi/180,100,minLineLength,maxLineGap)
    #print (lines)
    if(len(lines)):
        for i in xrange(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),7)
    '''

    prevTheta = 0.0
    lineColor = (0,0,255)
    lineThickness = 5
    count = 0
    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    if lines != None:
        #print len(lines)
        for i in xrange(len(lines)):
            for rho,theta in lines[i]:
                if abs(theta-prevTheta)>1:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    if x2 != x1:
                        m = (y2-y1)*1.0/(x2-x1)*1.0
                        if m>0.5 or m<-0.5 :
                            if count == 0:
                                lane1X1, lane1Y1, lane1X2, lane1Y2 = x1, y1, x2, y2
                                count += 1
                            elif count == 1 and y1>0:
                                lane2X1, lane2Y1, lane2X2, lane2Y2 = x1, y1, x2, y2
                                count += 1
                prevTheta = theta

    #print 'lane1: ', (lane1X1, lane1Y1), (lane1X2, lane1Y2)
    #print 'lane2: ', (lane2X1, lane2Y1), (lane2X2, lane2Y2)
    #print
    cv2.line(ROI, (lane1X1, lane1Y1), (lane1X2, lane1Y2), lineColor, lineThickness)
    cv2.line(ROI, (lane2X1, lane2Y1), (lane2X2, lane2Y2), lineColor, lineThickness)

    # Add edge detection part into original image
    image[offsetX:,:] = ROI
    return image

def Main():
    input_video =  'solidYellowLeft.mp4' #'VIDEO0024Trimmed.mp4' #'challenge.mp4' #'solidYellowLeft.mp4' #'solidWhiteRight.mp4'
    cap = cv2.VideoCapture(input_video)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Quit if either key 'q' is pressed or no frame available
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        # Process an image for lane detection
        processed_frame = LaneDetect(frame)

        # Display the resulting frame
        cv2.imshow('processed_frame',processed_frame)

if __name__ == '__main__' :
    Main()
