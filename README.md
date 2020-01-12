import cv2
from deepgaze.color_detection import MultiBackProjectionColorDetector

#Loading the main image
img = cv2.imread('tiger.jpg')

#Creating a python list and appending the model-templates
#In this case the list are preprocessed images but you
#can take subframes from the original image
template_list=list()
template_list.append(cv2.imread('model_1.jpg')) #Load the image
template_list.append(cv2.imread('model_2.jpg')) #Load the image
template_list.append(cv2.imread('model_3.jpg')) #Load the image
template_list.append(cv2.imread('model_4.jpg')) #Load the image
template_list.append(cv2.imread('model_5.jpg')) #Load the image

#Defining the deepgaze color detector object
my_back_detector = MultiBackProjectionColorDetector()
my_back_detector.setTemplateList(template_list) #Set the template

#Return the image filterd, it applies the backprojection,
#the convolution and the mask all at once
img_filtered = my_back_detector.returnFiltered(img, 
                                               morph_opening=True, blur=True, 
                                               kernel_size=3, iterations=2)
cv2.imwrite("result.jpg", img_filtered)