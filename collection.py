# Riley Karp
# CS442
# 5/10/2018
# collection.py

import numpy as np
import cv2 as cv
import sys
import csv
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# This method does nothing. It just needs to be passed to the createTrackbar()
# method in getLength()
def nothing(x):
    pass


# Takes in initial measured length between the blue tape markers
# Creates a video capture frame, detects the two blue markers,
# calculates the distance between the markers in each frame, and returns a list
# of the lengths. Starts calculating lengths when the toggle switch at the top
# of the window is turned on (set to 1). Video capture ends when 'q' is pressed.
# When 's' is pressed, program ends after saveing the lengths as a CSV file with
# filename 'serialNumber_length.csv', where the serial number is given in terminal
# window when prompted. Returns a list of the collected lengths.
def getLength( ):

    '''Initialize video capture and intial lengths'''
    cap = cv.VideoCapture(0)
    # len0 = float(measured_length) #original length in cm
    x0 = None #original pixel distance
    y0 = None
    positions = []
    times = []


    '''Create Toggle Switch To Record Length'''
    cv.namedWindow('original')
    switch = '0 : OFF , 1 : ON'
    cv.createTrackbar(switch, 'original',0,1,nothing)


    '''Find the 2 markers & calculate length between them in each frame'''

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:

            '''Find all blue contours in the frame'''
            # set region of interest (ROI) to middle rectangle of 400x200 pixels
            height = frame.shape[0]
            width = frame.shape[1]
            # ROI = frame
            ROI = frame[ (height/2)+200:height , 0:width ]

            # modify ROI and add to frame
            blurred = cv.GaussianBlur(ROI,(5,5),0)
            hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
            frame[ (height/2)+200:height , 0:width ] = hsv
            # frame = hsv

            # HSV Thresholds to identify blue chroma tape
            lower_blue_tape = np.array([110,20,20])
            upper_blue_tape = np.array([120,235,235])

            # create mask of all blue spots in the ROI & remove noise
            mask = cv.inRange(hsv, lower_blue_tape, upper_blue_tape)
            mask = cv.erode(mask, None, iterations=3)
            mask = cv.dilate(mask, None, iterations=3)
            # frame[ ((height/2)+40):(height/2)+200 , 0:width ] = mask

        	# find contours in the mask of identified blue spots in ROI
            conts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            conts = conts[1]


            '''Find and track the marker'''
    		# find the largest blue contours in the ROI
            if len(conts)>=1:
                conts.sort(key=cv.contourArea)
                box1 = conts[-1]

                # create bounding rectangles and centroids around the 2 largest contours
                x1,y1,h1,w1 = cv.boundingRect(box1)
                # print( h1, w1 )

                center = ( x1+(h1/2) , y1+(w1/2) )


        		# only proceed and draw the rectangles if they arae a specified size
                '''May need to adjust sizes in if statement based on experimental setup.
                Adjust by printing h1,h2,w1,w2 to find desired contour sizes.
                Some commonly used boundaries can be seen commented out below'''
                # print h1,w1
                # if 30<w1<50 and 12<h1<30:
        			# draw the rectangles and centroids on the frame
                cv.rectangle( ROI, (int(x1),int(y1)), (int(x1+h1),int(y1+w1)), (0, 255, 0), 1 )
                cv.circle( ROI, center, 5, (0, 0, 255), -1)


                '''Find the vertical distance between the 2 markers'''
                on = cv.getTrackbarPos(switch,'original')
                if on == 1:
                    # initialize dist0 if it's the first measurement frame
                    if x0 == None:
                        x0 = center[0]
                        y0 = center[1]
                        start = time.time()

                    # find the pixel distance between the inner edges of the 2 rectangles
                    pos = (center[0]-x0,center[1]-y0)
                    end = time.time()
                    t = (end-start)
                    times.append( t ) # seconds

                    # calculate actual length using initial measured lengths
                    # length = pix * len0 / dist0
                    # print length
                    print(t, pos)
                    positions.append(pos)

            '''Display frames'''
            cv.imshow('mask',mask) # black and white showing all blue items in ROI
            cv.imshow('original', frame)


            '''Quit if q is pressed'''
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            # save and quit if s is pressed
            elif cv.waitKey(1) & 0xFF == ord('s'):
                filename = raw_input('Type foot ID: ')
                save(times,positions, filename+'_data')
                break
        else:
            break

    # close all windows and end program
    cap.release()
    cv.destroyAllWindows()

    return times, positions


# Takes in a list of data and a filename string and saves the data as a CSV file
# with the given filename.
def save( times,lengths, filename, plats = False ):
    file = filename + '.csv'
    f = open( file, 'wb' )
    writer = csv.writer( f, delimiter=',', quoting=csv.QUOTE_MINIMAL )
    for i in range(len(times)):
        writer.writerow([times[i],lengths[i][0],lengths[i][1]])
    f.close()


# Takes in a CSV filename. Assumes the data is in the form produced by the
# getLength method, where the file only contains one column (of lengths)
# Returns a list of the lengths.
def read( filename ):

    times = []
    xpos = []
    ypos = []
    file = open( filename, 'rU' )
    rows = csv.reader(file)
    for row in rows:
        times.append( float(row[0]) )
        xpos.append( float(row[1]) )
        ypos.append( float(row[2]) )
    file.close()

    return times,xpos,ypos

def fixX(xpos):
    return [ -x/8 for x in xpos ]

def fixY(ypos):
    return [ y/8 for y in ypos ]

def plotTrials(args):

    curv = args[1]
    trials = int(args[2])

    fig, axs = plt.subplots(2,1)

    # fcurv_triali_data.csv
    for i in range(1,trials+1):
        file = curv+"_trial"+str(i)+"_data.csv"
        times,xpos,ypos = read(file)
        axs[0].plot(times,fixX(xpos),label="trial{0}".format(i))
        axs[1].plot(times,fixY(ypos))

    axs[0].set_title("X vs. Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("x-position (cm)")
    axs[0].legend(loc='center left')

    axs[1].set_title("Y vs. Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("y-position (cm)")

    fig.tight_layout()

    plt.show()

def plotOne(file):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig, axs = plt.subplots(2,1)

    times,xpos,ypos = read(file)
    # axs[0].plot(times,fixX(xpos))
    # axs[1].plot(times,fixY(ypos))

    # axs[0].plot(fixX(xpos),fixY(ypos))
    ax.plot(times,fixY(ypos))

    # axs[0].set_title("X vs. Time")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("x-position (cm)")

    ax.set_title("Y vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("y-position (cm)")

    fig.tight_layout()

    plt.show()

def exp_func( x, a, b, c ):
    return a*np.exp(-b*x)+c

def plotAvg():
    # efficiency = time / volume
    fig = plt.figure()
    ax = fig.add_subplot(111)

    curv = [ 1.0/5, 1.0/10, 1.0/20, 1.0/30, 1.0/40 ] #cm-1
    volume = [ 27.5, 16.5, 7.0, 5.0, 3.5 ] #cm^3
    time = [ [ 39.6, 41.0, 54.2, 50.6, 35.9 ],
             [ 44.6, 30.0, 28.9, 29.6, 35.6 ],
             [ 37.4, 31.3, 28.3, 39.9, 28.9 ],
             [ 42.4, 34.4, 25.7, 36.7, 25.1 ],
             [ 29.7, 28.9, 28.7, 28.2, 29.9 ] ] # s

    ranges = [ [ 16.625, 7.125, 5.5, 14.0, 6.875 ],
             [ 10.875, 5.5, 5.125, 5.25, 5.75 ],
             [ 8.75, 15.5, 17.625, 9.125, 18.125 ],
             [ 5.25, 8.5, 4.75, 7.0, 5.75 ],
             [ 5.25, 5.625, 10.125, 5.125, 4.875 ] ] # s

    curvs = []
    effs = []
    for i in range( len(curv) ):
        for j in range( len(time[i]) ):
            curvs.append(curv[i])
            effs.append(time[i][j]/volume[i])
            # effs.append(ranges[i][j])

    popt, pcov = curve_fit( exp_func, curvs, effs, p0=(1, 1e-6, 1) )
    print(popt)
    xs = np.linspace(0.01,0.25,100)
    ys = exp_func(xs,*popt)
    ax.plot(xs,ys,'-')

    ax.plot(curvs[0:5],effs[0:5],'.',label="0.200")
    ax.plot(curvs[5:10],effs[5:10],'.',label="0.100")
    ax.plot(curvs[10:15],effs[10:15],'.',label="0.050")
    ax.plot(curvs[15:20],effs[15:20],'.',label="0.033")
    ax.plot(curvs[20:],effs[20:],'.',label="0.025")

    ax.legend(loc='upper right')

    ax.set_title("Efficiency vs. Curvature")
    ax.set_xlabel("Curvature (1/cm)")
    ax.set_ylabel("Efficiency (s/cm^3)")

    plt.show()

def getRanges(args):

    curv = args[1]
    trials = int(args[2])

    # fcurv_triali_data.csv
    for i in range(1,trials+1):
        file = curv+"_trial"+str(i)+"_data.csv"
        time, xs, ys = read(file)
        ys = fixY(ys)
        print("trial " + str(i) + ": " + str( max(ys)-min(ys) ))

def rmse( predictions, targets ):
    return np.sqrt(sum([ (x-y)**2 for x,y in zip(predictions,targets) ])/len(predictions) )

def sin_func(x, a, b, c):
    return [ (a*np.sin( b*i ) + c) for i in x ]

def sin_fit_func(x, a, b, c):
    return ((a*np.sin( b*x )) + c)

def sinFit(file):
    # choose trial w time closest to average
    # constant A, vary f, minimize RMSE
    times, xs, ys = read(file)
    ys = fixY(ys)

    a = 2
    c = 2
    minRMSE = 100000
    bestF = 0
    bestFit = ys
    f = 0.1
    while f < 100:
        fit = sin_func(times,a,f,c)
        err = rmse(fit,ys)
        # print err
        if( err < minRMSE ):
            minRMSE = err
            bestF = f
            bestFit = fit
        f += 0.1

    print "RMSE: " , minRMSE
    print "Period: ", bestF
    # print "Fits: " , bestFit

    fig = plt.figure()
    ax = fig.add_subplot(111)

    popt, pcov = curve_fit( sin_fit_func, times, bestFit )
    print(popt)
    fitxs = np.linspace(0,45,425)
    fitys = sin_fit_func(np.matrix(times),*popt)

    ax.plot(times,ys,"-",label="Experimental Data")
    ax.plot(times,bestFit,".",label="Sinusoidal fit")
    ax.plot(fitxs,fitys.T,label="Sinusoidal Curve Fit")
    ax.legend(loc='upper left')

    ax.set_title("Y-position vs. Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y-position (cm)")
    ax.set_ylim([-2,6])

    plt.show()

if __name__ == '__main__':
    # times, lengths = getLength()
    # print(times)
    # print(lengths)
    # plotTrials(sys.argv)
    plotAvg()
    # plotOne(sys.argv[1])
    # sinFit( sys.argv[1])
    # getRanges(sys.argv)
