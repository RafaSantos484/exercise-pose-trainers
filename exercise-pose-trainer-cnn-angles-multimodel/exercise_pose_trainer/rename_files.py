import sys
import os
import uuid


def rename_files_random(folder: str):
    if not os.path.isdir(folder):
        print("Invalid folder path")
        return

    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        if os.path.isfile(filepath):
            _, extension = os.path.splitext(file)
            new_name = f"{uuid.uuid4().hex}{extension}"
            new_path = os.path.join(folder, new_name)

            os.rename(filepath, new_path)
            print(f"{file} -> {new_name}")


def rename_files_sequential(folder: str):
    if not os.path.isdir(folder):
        print("Invalid folder path")
        return

    files = [f for f in os.listdir(
        folder) if os.path.isfile(os.path.join(folder, f))]
    files.sort()

    total_files = len(files)
    # Define o número de dígitos com base na quantidade de arquivos
    padding = len(str(total_files))

    for i, file in enumerate(files, start=1):
        filepath = os.path.join(folder, file)
        _, extension = os.path.splitext(file)
        new_name = f"{str(i).zfill(padding)}{extension}"
        new_path = os.path.join(folder, new_name)

        os.rename(filepath, new_path)
        print(f"{file} -> {new_name}")


def main(args=sys.argv[1:]):
    if len(args) == 0:
        print("Missing folder path parameter")
        return

    """
    if len(args) > 1 and args[1] in ["seq", "sequential"]:
        rename_files_sequential(args[0])
    else:
        rename_files_random(args[0])
    """
    rename_files_sequential(args[0])
