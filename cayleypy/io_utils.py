import json


# disabling pylint warning about explicitly specifying the encoding
# pylint: disable=W1514
def write_json(path, obj):
    with open(path, "w+") as file:
        json.dump(obj, file, indent=4)


def read_json(path):
    with open(path, "r") as file:
        return json.load(file)


def write_txt(path, obj):
    assert isinstance(obj, str), "Only strings are accepted for txt writing"
    with open(path, "w+") as file:
        file.write(obj)


def read_txt(path):
    with open(path, "r") as file:
        return file.read()


# pylint: enable=W1514
