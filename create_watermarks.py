import os
import cv2
from imutils import paths
import numpy as np
import math
import re


if __name__ == "__main__":

    '''
        Input arguments : fill in what we want below
    '''

    input_folder = 'images'             # path to the folder where the images we want to watermark are located
    output_folder = 'watermarked_images'    # path to the folder
    watermark_img = 'logo_test.png'  # path to the watermark

    wm_type = 4                   # 0 = top left, 1 = top right, 2 = bottom left, 3= bottom right, 4=middle
    wm_scale = 1                 # scale of the watermark
    wm_opacity = 0.3             # opacity of the watermark over the image

    wm_name = '_wm'             # what we want to append to the name of the images when saving in the output folder
                                # leave empty if we want it to have the same name
    progress_bar = True         # if set to True, will show a progress bar whilst loading and watermarking

    # load the watermarked image
    watermark = cv2.imread(watermark_img, cv2.IMREAD_UNCHANGED)

    # Change the size of the watermark based on th scale of the thing
    watermark_height = int(watermark.shape[0] * wm_scale)   # height of our watermark
    watermark_width = int(watermark.shape[1] * wm_scale)    # width of our watermark
    watermark = cv2.resize(watermark, (watermark_width, watermark_height), interpolation=cv2.INTER_LINEAR)

    # add alpha channel if it doesn't exist
    layers_n = watermark.shape[2]
    if layers_n == 3 : # missing alpha layer in it layer
        (B, G, R) = cv2.split(watermark)
        A = np.ones(B.shape, dtype=B.dtype) * 0  # Fake alpha channel which we can give to the png
        watermark = cv2.merge((B, G, R, A))

    # Convert each of the channels to bitwise, which allows us to have transparent pixels
    (B, G, R, A) = cv2.split(watermark)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    watermark = cv2.merge([B, G, R, A])

    # Progress bar:
    if progress_bar:
        pb_c = 0                                            # how many of the images we have done
        pb_t = len(list(paths.list_images(input_folder)))   # how many there are total
        print("We have started the watermarking")
        print("Total number of images to watermark : {}".format(pb_t))

    # loop over the images in the input folder, and watermark each o fthem
    for imagePath in paths.list_images(input_folder):
        image = cv2.imread(imagePath)   # read the image we want to watermark
        image_height = image.shape[0]
        image_width = image.shape[1]

        # get the transparancy value (255 * ones)
        image = np.dstack([image, np.ones((image_height, image_width), dtype="uint8") * 255])

        # create the overlay
        overlay = np.zeros((image_height, image_width, 4), dtype="uint8")

        # depending on type, find the x1, y1, x2, y2 of where we ideally would want the watermark to go
        if wm_type == 0:       # top left
            loc = [10, 10, 10 + watermark_width, 10 + watermark_height]
        elif wm_type == 1:     # top right
            loc = [image_width - 10 - watermark_width, 10, image_width - 10, 10 + watermark_height]
        elif wm_type == 2:     # bottom left
            loc = [10, image_height - 10 - watermark_height, 10 + watermark_width, image_height - 10]
        elif wm_type == 3:     # bottom right
            loc = [image_width - 10 - watermark_width, image_height - 10 - watermark_height, image_width - 10,
                   image_height - 10]
        elif wm_type == 4:     # middle
            loc = [math.ceil(image_width / 2 - watermark_width / 2), math.ceil(image_height / 2 - watermark_height / 2),
                   math.floor(image_width / 2 + watermark_height / 2), math.floor(image_height / 2 + watermark_height / 2)]
        else:   # bottom right as default
            loc = [image_width - 10 - watermark_width, image_height - 10 - watermark_height, image_width - 10,
                   image_height - 10]

        # check if our x1,y1,x2,y2 are out of bounds. If so, change it so that it is within bounds
        loc = [max(0, loc[0]), max(0, loc[1]), min(image_width, loc[2]), min(image_height, loc[3])]

        # resize watermark to fit in the image (so it doesn't exceed the size of the image)
        watermark_to_apply = cv2.resize(watermark, (loc[2] - loc[0], loc[3] - loc[1]), interpolation=cv2.INTER_LINEAR)
        overlay[loc[1]:loc[3], loc[0]:loc[2]] = watermark_to_apply

        # blend together with cv2.addWeighted
        output = image.copy()
        cv2.addWeighted(overlay, wm_opacity, output, 1.0, 0, output)

        # Save the image under the name which includes the extension
        filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
        result = re.split("\.", filename)                                   # split by period so we can add extension
        new_name = result[0]+wm_name+"."+result[1]                 # this is the name under which we save
        destination_location = os.path.sep.join((output_folder, new_name))
        cv2.imwrite(destination_location, output)

        # Show progress bar
        if progress_bar:
            pb_c += 1
            toadd = round(pb_c/pb_t*50)
            todisplay = "Progress : ["+ toadd*"-"+">"+ (50-toadd)*" "+"] {}/{}".format(pb_c, pb_t)
            print(todisplay)
