import PIL
from PIL import Image

# Code source: https://gist.github.com/tomvon/ae288482869b495201a0
# TODO: how to make this method as a class? Pass args by constructor and how to output the image itself just by returning the class object? (like Image(filename) does)
"""
class ImageResizer(Image):
    def __init__(self, filename: str, resizeBy: float = 0.75):
        img = Image.open(filename)

        newWidth = int(resizeBy * img.size[0])
        newHeight = int(resizeBy * img.size[1])
        newImg = img.resize((newWidth, newHeight), PIL.Image.ANTIALIAS)

        self.resizedImage = newImg

    def __str__(self):
        self.resizedImage
"""

def resize(filename: str, resizeBy: float = 0.75) -> Image:
    img = Image.open(filename)

    newWidth: int = int(resizeBy * img.size[0])
    newHeight: int = int(resizeBy * img.size[1])
    newImg = img.resize((newWidth, newHeight), PIL.Image.ANTIALIAS)

    return newImg
