from PIL import Image
import numpy as np
import sys
import os
import csv

# default format can be changed as needed
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

"""
encoding the asl aplphabet
0 - 25: A - Z
26: del
27: nothing
28: space  
"""
def encodeType(filename):
    encoding = None
    first_letter = filename[0]
    match first_letter:
        case "A":
            encoding = 0
        case "B":
            encoding = 1
        case "C":
            encoding = 2
        case "D":
            encoding = 3
        case "E":
            encoding = 4
        case "F":
            encoding = 5
        case "G":
            encoding = 6
        case "H":
            encoding = 7
        case "I":
            encoding = 8
        case "J":
            encoding = 9
        case "K":
            encoding = 10
        case "L":
            encoding = 11
        case "M":
            encoding = 12
        case "N":
            encoding = 13
        case "O":
            encoding = 14
        case "P":
            encoding = 15
        case "Q":
            encoding = 16
        case "R":
            encoding = 17
        case "S":
            encoding = 18
        case "T":
            encoding = 19
        case "U":
            encoding = 20
        case "V":
            encoding = 21
        case "W":
            encoding = 22
        case "X":
            encoding = 23
        case "Y":
            encoding = 24
        case "Z":
            encoding = 25
        case "d":
            encoding = 26
        case "n":
            encoding = 27
        case "s":
            encoding = 28
    return encoding


# load the original image
myFileList = createFileList('/scratch/rvelasc1/Downloads/asl_archive/asl_alphabet_test/asl_alphabet_test')

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    encoding = encodeType(os.path.basename(file))
    print(f"file: {file}, encoding: {encoding}")
    print(value)
    with open('/scratch/rvelasc1/asl_alphabet_test.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

    # build answer file
    with open('/scratch/rvelasc1/asl_alphabet_test_ans.txt', 'a') as g:
        g.write(f"{encoding}\n")
