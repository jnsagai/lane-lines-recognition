# **Finding Lane Lines on the Road** 

## Writeup Template
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[original_image]: ./examples/original_image.jpg "Original Image"
[gamma_image]: ./examples/gamma_image.jpg "Gamma Image"
[gray_image]: ./examples/gray_image.jpg "Gray Image"
[blur_image]: ./examples/blur_image.jpg "Blur Image"
[canny_image]: ./examples/canny_image.jpg "Canny Image"
[masked_edges]: ./examples/masked_edges.jpg "Masked Image"
[lines_image]: ./examples/lines_image.jpg "Lines Image"
[line_image]: ./examples/line_image.jpg "Line Image"
[combo_image]: ./examples/combo_image.jpg "Combo Image"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][original_image]

My pipeline consisted of 10 steps:
1. I read the original image a create a safety copy of it. An example of an original image with the lane lines follows below:
![alt text][original_image]

2. In the next step I apply a method of [Gamma correction](https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html) in the original image, in order to highlight the differences between the road and the lines. This extra step was created in order to accomplish the optional Challenge. :)
![alt text][gamma_image]

3. It is much easier and computationally efficient to process the image in grayscale color space. Thus the next step is the color conversion:
![alt text][gray_image]

4. After that a blur effect is applied through a Gaussian Smoothing:
![alt text][blur_image]

5. Now it is time to find the gradient points (edges) in the image using [Canny Edge Detection](https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html):
![alt text][canny_image]

6. As you can see in the previous image, the Canny algorithm was able to find several edges points throughout the image. However we only care about those points related to the lane lines. The next step consists of applying a mask using binary filtering to "erase" (set the pixels to 255) all the non-relevants points, keeping just those points inside the region of interest. This region is created by a polygon that surrounds the lane lines. The result can be seen below:
![alt text][masked_edges]

7. In this step we identify those points whose pattern could represent a line. In order to do so, we use the Hough Transform to aggregate the points into line:
![alt text][lines_image]

8. Now we have several detected lines on the right and left lanes. However its more useful if we'd have just one single line on each side that represents the lane, also in terms of further processing (keep the car into the lanes). I accomplish that by modifying the draw_lines() function where I first identified and grouped the lines of each side (base on the slope of the line) and then calculated the average of the line parameters (slope and intercept) in order to create a single line. The first point of this new line is defined using the origin of the first line in the group and the second point using the limit of our region of interest (See Step 6).
![alt text][line_image]

9. Finally we can join the original image with the calculated lines as the process outcome:
![alt text][combo_image]

### 2. Identify potential shortcomings with your current pipeline

One shortcoming that was identified was that some noise still persisted even after the Canny and the Hough Transform. Some of the line noises had slope value far from average, which led to a distortion in the final lane line. This effect was mitigated by applying a filter in the line slope calculation, that means, just a line within a defined slope range was accepted. The range boundary was estimated based on the log of several lines when running the code with the videos.

Another shortcoming was that some small lines were identified as being from one lane side based on its slope however it was presented on the opposite side. To fix it I created a virtual point right in the middle x-axis of the region of interest. If such abnormal line appears it would be discarded right away base on its position.

Another shortcoming was related to the challenge, where my original algorithm was not able to deal with the region where the black asphalt disappears and then a concrete road takes place. I have to use a Gamma correction filter to highlight even more the lane lines and surpass this situation.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to better tune the parameters of the algorithms to enhance their robustness. Maybe I should use other methods to highlight even more the lane lines over the road.

Another potential improvement could be to apply a digital filter over lane lines calculation between each video frame in order to smooth the visualization and reduce the bouncing of the lines.
