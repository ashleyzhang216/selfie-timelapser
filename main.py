import os

ALL_DATA_PATH = "data/"
RAW_DATA_PATH = "raw/"
CONVERTED_DATA_PATH = "png/"

def make_dirs():
    if not os.path.exists(ALL_DATA_PATH):
        os.makedirs(ALL_DATA_PATH)

    if not os.path.exists(ALL_DATA_PATH + RAW_DATA_PATH):
        os.makedirs(ALL_DATA_PATH + RAW_DATA_PATH)
    if not os.path.exists(ALL_DATA_PATH + CONVERTED_DATA_PATH):
        os.makedirs(ALL_DATA_PATH + CONVERTED_DATA_PATH)

if __name__ == "__main__":
    make_dirs()
    print("Hello world!")
