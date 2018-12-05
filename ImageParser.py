
import os
import cv2
import re
import numpy as np
import json
import pprint
def load(dirname, savedir):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    subdir_path = dir_path + '/' + dirname

    for subdir in os.listdir(subdir_path):
        filedir_path = subdir_path + '/' + subdir
        savepath = dir_path + '/' + savedir + '/' + subdir
        os.mkdir(savepath)
        #print(filedir_path)
        for filename in os.listdir(filedir_path):
            filestr = filename
            filestr = filestr.split("_")
            birth = filestr[1].split("-")
            yeartaken = filestr[2].split(".")
            age = int(yeartaken[0]) - int(birth[0])
            #print(age)
            newfilename = filestr[0] + "." + str(age) + ".jpg"
            #print(filestr)
            file_path = filedir_path + '/' + filename

            print(file_path)
            pic = cv2.imread(file_path)
            #print(pic)
            #np.array(pic)
            pic = cv2.resize(pic, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)


            savefilepath = dir_path + '/' + savedir + '/' + subdir + '/' + newfilename
            print(savefilepath)
            cv2.imwrite(savefilepath, pic)
            print(pic.shape)

'''
check if it is jsut an black and white image
'''
def isImgVaild(pic):
    for i in range(120):
        for j in range(120):
            if pic[i][j] != 0 and pic[i][j] != 255:
                return True
    return False
'''
rootdir: the root directory of the input folder
filename: the .txt file to same
'''
def parse(rootdir, filename, count):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    subdir_path = dir_path + '/' + rootdir
    save_path = dir_path + '/' + "Json"

    c = 0
    for subdir in os.listdir(subdir_path):
        ds = {}
        filedir_path = subdir_path + '/' + subdir
        # print(filedir_path)
        if c == count:
            print("saving...")
            with open("data.json", 'w') as outfile:
                json.dump(ds, outfile)
            return

        for filename in os.listdir(filedir_path):
            file_path = filedir_path + '/' + filename
            print(file_path)
            pic = cv2.imread(file_path, 0)
            #print(pic)
            filename = filename.split(".")
            age = int(filename[1])
            if isImgVaild(pic) and age <= 100 and age >= 1:
                ds.update({filename[0]: {"age": age,
                                        "img": pic.tolist()}
                        })
        savefile_path = "Json" + "/" + str(c) + ".json"
        with open(savefile_path, 'w') as outfile:
            json.dump(ds, outfile)
        c+=1




def read_json():
    print("reading...")
    with open('data.json') as f:
        data = json.load(f)

    for d in data:
        print(d)


#load("wiki_crop", "wiki")
parse('wiki', 'data.json', 100)
#read_json()