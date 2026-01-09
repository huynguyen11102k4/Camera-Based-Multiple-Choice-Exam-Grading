import os
from pipeline import imagePineline

if __name__ == "__main__":
    root = os.getcwd()
    imgPath = os.path.join(root, 'images/i1.jpg')
    imagePineline(imgPath)