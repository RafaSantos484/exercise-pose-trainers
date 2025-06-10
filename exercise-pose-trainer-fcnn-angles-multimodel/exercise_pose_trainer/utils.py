import os


def is_img_file(path: str):
    image_extensions = ['.jpg', '.jpeg', '.png',
                        '.gif', '.bmp', '.tiff', '.webp']
    return any(path.lower().endswith(ext) for ext in image_extensions)


def get_basename(path: str):
    return os.path.basename(os.path.normpath(path))
