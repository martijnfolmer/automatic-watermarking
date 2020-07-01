# Automatic watermarking of images

In this script, we automatically copy and append images with a watermark. 

To use this script, you must run create_watermarks.py. There are several variables (line 19- line 29) that you can change in order to suit your needs. These are:

- input_folder : The path to the folder in which we have stored the images we wish to watermark
- output_folder : The path to the folder in which we want to store the final watermarked images
- watermark_img : The path to the image we are going to use as a watermark. For the best experience, you should use a .png
- wm_type       : This is an integer between 0 and 4 dictates where the watermark is located in the image. The differences can be seen in the below image
![type_image](readme_img/wm_type.jpg)

- wm_scale     : This if a float and dictates if you want to resize the watermark. This can be seen in the below image
![scale_image](readme_img/wm_scale.jpg)

- wm_opacity   : this is a float between 0 and 1 and dictates how transparent your watermark image will be. This can be seen in the below image
![opacity_image](readme_img/wm_opacity.jpg)

- wm_name     : This is a string which will be appended to the end of the filename of the image want to watermark, which will be used as the new name for our watermarked image. Leave empty if you want the watermarked images to have the same name

- progress_bar   : This is a boolean. If set to True, a progress bar will be printed whilst the images are being watermarked to show how far along you are.


In the future, I plan on creating a similar script that allows for watermarking of videos.
