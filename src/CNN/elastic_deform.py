import numpy, imageio, elasticdeform
from matplotlib import image
import os
import numpy as np




class elastic_deform():
    
    """
    elastic  deformation on cell image. implemented from pythorch package 
    pip install git+https://github.com/gvtulder/elasticdeform
    motivated by U-net original paper
    
    Input: image folder: name of the folder containing traing data that need to be augmented
           sigma       : parmeters for augmentation. default:3
           points      : parmeters for augmentation. default:2
           
    Output: saves the augmented images on /augmented_images/ which is a sister directory of image_folder "
    """

    def __init__(self, image_folder) -> None:
        
        self.image_folder = image_folder
        self.augmented_images = os.path.abspath(os.path.join(image_folder, os.pardir)) + "/augmented_images/"
        
        
    def augment(self, sigma=3, points=2):
        if not os.path.exists(self.augmented_images):
                os.makedirs(self.augmented_images)
        for folder in os.listdir(self.image_folder):
            current_folder = os.path.join(self.image_folder, folder)
            augmented_folder = self.augmented_images + folder + "/"
            
            if not os.path.exists(augmented_folder):
                os.makedirs(augmented_folder)
            for image_name in os.listdir(current_folder):
                current_image = os.path.join(current_folder,  image_name)
                my_image = np.array(image.imread(current_image))
                images_deformed = elasticdeform.deform_random_grid(my_image, sigma=3, points=2)
                imageio.imsave( augmented_folder + image_name.split(".")[0] + "_aug.tif", images_deformed)
            

    def copy_aug(self):
        import shutil
        for folder in os.listdir(self.image_folder):
            current_folder = os.path.join(self.image_folder, folder)
            augmented_folder = self.augmented_images + folder + "/"
            for image_name in os.listdir(augmented_folder):
                current_image = os.path.join(augmented_folder,  image_name)
                shutil.copy(current_image, current_folder)