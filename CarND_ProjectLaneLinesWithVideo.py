"""
Created on Sat Aug  1 22:00:29 2020

@author: Jefferson Nascimento
"""

# Importing some useful packages
import numpy as np
import cv2

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

##############################################################################
#########################   GLOBAL PARAMETERS    #############################
##############################################################################

# Test Videos
test_video_input_challenge = 'test_videos/challenge.mp4'
test_video_input_solid_white = 'test_videos/solidWhiteRight.mp4'
test_video_input_solid_yellow = 'test_videos/solidYellowLeft.mp4'

# Video Output
video_output_challenge = 'test_videos_output/challenge.mp4'
video_output_solid_white = 'test_videos_output/solidWhiteRight.mp4'
video_output_solid_yellow = 'test_videos_output/solidYellowLeft.mp4'

# Canny parameters
canny_low_threshold = 50
canny_high_threshold = 150

# Define the Hough transform parameters
ρ = 1                   # distance resolution in pixels of the Hough grid
θ = np.pi/180           # angular resolution in radians of the Hough grid
threshold = 20          # minimum number of votes (intersections in Hough grid cell)
min_line_lenght = 15    # minimum number of pixels making up a line
max_line_gap = 3        # maximum gap in pixels between connectable line segments

# Define min and max slope per line side
left_min_slope = -0.84
left_max_slope = -0.64
right_min_slope = 0.53
right_max_slope = 0.72

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)    
 
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, line, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_image = np.zeros_like(img)
    if line is not None:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    
    return line_image

def hough_lines_image(img, ρ, θ, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, ρ, θ, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hough_lines(img, ρ, θ, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns the hough lines.
    """
    lines = cv2.HoughLinesP(img, ρ, θ, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def calc_coordinates(img, line_parameters, top_coord_line):
    """    
    Parameters
    ----------
    img : image
        base image
    line_parameters : tuple
        line linear parameters ( m and b).
    top_coord_line : int32
        line top coordinate ( y2 ).

    Returns
    -------
    Array [x1,y1,x2,y2] 
    """
   
    m, b = line_parameters
    # Get the height of the image as y1 (Remember that the image origin is on the top left corner)
    y1 = img.shape[0]
    # Get the top coordinate as y2
    y2 = top_coord_line
    # Calc x1 and x2 using the linear equation: x = (y - b) / m
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    
    return np.array([x1, y1, x2, y2])

def average_side_lines(img, lines, top_coord_line, center_line):
    """    
    Parameters
    ----------
    img : image
        original image.
    lines : lines
        array of lines.
    top_coord_line : int32.
        line top coordinate ( y2 )
    center_line : int32.
        line used to separate right and left lines

    Returns
    -------
    Array with right and left side lines
    """
    
    left_avg_line = []
    right_avg_line = []
    
    # Classify the lines between left and right according to their slope
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Find the polinomial parameters of the curse, in this case
        # the slope (m) and the intercept (b) since it is a linear one
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # Check whether it is a right side or left side line and store it
        # in the right list
        if (slope >= left_min_slope and slope <= left_max_slope) and (x1 < center_line and x2 < center_line):
            left_avg_line.append((slope, intercept))
        elif (slope >= right_min_slope and slope <= right_max_slope) and (x1 > center_line and x2 > center_line):
            right_avg_line.append((slope, intercept))        
   
    # Calculate the average value of the lines in the list with respect
    # to the vertical axis. Then calculate the coordinates for a fit line
    left_line = np.zeros(4, dtype=int)
    right_line = np.zeros(4, dtype=int)
    
    # Check whether there are lines on each side
    if left_avg_line:        
        left_fit_average = np.average(left_avg_line, axis=0)
        left_line = calc_coordinates(img, left_fit_average, top_coord_line)
    
    if right_avg_line:
        right_fit_average = np.average(right_avg_line, axis=0)
        right_line = calc_coordinates(img, right_fit_average, top_coord_line)
    
    return np.array([left_line, right_line])

    
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def apply_brightness_and_contrast(image, α, β):    
    
    new_image = cv2.convertScaleAbs(image, alpha=α, beta=β)        
    return new_image

def gamma_correction(image, γ):
    lookUpTable = np.empty((1,256), np.uint8)
    
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, γ) * 255.0, 0, 255)
    
    new_image = cv2.LUT(image, lookUpTable)
    return new_image

##############################################################################
##############################   MAIN CODE   #################################
##############################################################################

def process_image(image):
    
    # Make a copy of the original image in order to not modify it
    lane_image =  np.copy(image)
    
    gamma_image = gamma_correction(lane_image, 3)
    
    # Apply the gray scale conversion at the image
    gray_image = grayscale(gamma_image)
    
    # Define a kernel size and apply Gaussian Smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray_image, kernel_size)    
  
    # Apply Canny algorithm
    canny_image = canny(blur_gray, canny_low_threshold, canny_high_threshold)
    
    # Create a masked edges
    # Define a four sided polygon to mask
    #      _____
    #     /     \
    #    /       \
    #   /         \
    #  /           \
    # /_____________\
    #
    imshape = image.shape
    # Define vertices coordinates
    ver_1_x = int((imshape[1] / 14))
    ver_1_y = int(imshape[0])
    ver_2_x = int((imshape[1] / 2) - (imshape[1] / 8))
    ver_2_y = int((imshape[0] * 5) / 8)
    ver_3_x = int((imshape[1] / 2) + (imshape[1] / 8))
    ver_3_y = int(ver_2_y)
    ver_4_x = int(imshape[1] - (imshape[1] / 14))
    ver_4_y = int(imshape[0])
    
    # Define the center line x coordinate used for separate right lines from left lines
    # In this case it is placed in the center of the polygon
    center_line_coord = ver_2_x + ((ver_3_x - ver_2_x) / 2)
    
    vertices = np.array([[(ver_1_x, ver_1_y), (ver_2_x, ver_2_y), (ver_3_x, ver_3_y), (ver_4_x, ver_4_y)]], dtype=np.int32)
    masked_edges = region_of_interest(canny_image, vertices)
    
    # Identify the lines in the image through Hough Transform algorithm
    # Output lines is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, ρ, θ, threshold, min_line_lenght, max_line_gap)
    
    # Instead of showing all the lines in the right and left side
    # it is better to show just an average of the lines
    average_lines = average_side_lines(lane_image, lines, ver_2_y, center_line_coord)
    
    # Get a image with the detected lines
    line_image = draw_lines(lane_image, average_lines)
    
    # Draw the lines on the original image
    combo_image = weighted_img(line_image, lane_image)
    
    return combo_image

clip1 = VideoFileClip(test_video_input_challenge)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(video_output_challenge, audio=False)

clip1 = VideoFileClip(test_video_input_solid_white)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(video_output_solid_white, audio=False)

clip1 = VideoFileClip(test_video_input_solid_yellow)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(video_output_solid_yellow, audio=False)
