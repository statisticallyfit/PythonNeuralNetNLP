# Importing the image related things:
import sys
import os
import IPython.display
import ImageResizer
# from IPython.display import Image

# Making files in utils folder visible here:
#
# import ImageResizer

# Building pathname for images
def setImagePath() -> str:
    # Making files in utils folder visible here:
    sys.path.append(os.getcwd() + "/src/utils/")

    pth: str = os.getcwd()
    imagePath: str = pth + "/src/ModelStudy/images/"
    return imagePath

__all__ = ['sys', 'os', 'IPython.display', 'ImageResizer']
