"""
    convert png to jpg and compress, used for latex & overleaf so compiles faster and creates smaller pdf's
"""
from PIL import Image
from os import remove
from glob import glob
from os.path import join


def compress_png(image) -> None:
    img = Image.open(image).convert("RGB")

    # save image as jpeg in order to reduce size, can further be compressed by adding a 'quality' kwarg to save()
    image = image.replace("png", "jpg")
    img.save(image, "JPEG", optimize=True)


if __name__ == "__main__":
    # take all png in the figures directory and all subdirectories (basically every png in latex directory)
    file_path = join(r"..", "..", "latex_sa_da", "figures", "**", "*.png")

    # compress all found png to jpg
    [compress_png(f) for f in glob(file_path, recursive=True)]

    # then delete the old png files
    [remove(f) for f in glob(file_path, recursive=True)]
