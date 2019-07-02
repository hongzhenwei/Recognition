from aip import AipImageClassify #安装baidu库导入AipImageClassify类
import json
""" 你的 APPID AK SK """
APP_ID = '16599142'

API_KEY = 'TncfhZ0ETx4p0XslhRwUNnne'

SECRET_KEY = '5pVteFHgEKUKH7aWvIadLwfckgFhgRDe'

def myDishDetect():

    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    """ 读取图片 """
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    image = get_file_content('static/images/test.jpg')

    """ 调用菜品识别 """

    """ 如果有可选参数 """
    options = {}
    options["top_num"] = 3
    options["filter_threshold"] = "0.7"
    options["baike_num"] = 5

    """ 带参数调用菜品识别 """
    result=client.dishDetect(image, options)

    print(json.dumps(result))
    return result #返回的是字典对象

# 动物识别调用aip
def myanimalDetect():

    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    """ 读取图片 """
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    image = get_file_content('static/images/test.jpg')

    """ 调用动物识别 """

    """ 如果有可选参数 """
    options = {}
    options["top_num"] = 3
    #options["filter_threshold"] = "0.7"
    options["baike_num"] = 5

    """ 带参数调用动物识别 """
    result=client.animalDetect(image, options)
    print(json.dumps(result))
    return result #返回的是字典对象

# 植物识别调用aip
def myplantDetect():

    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

    """ 读取图片 """
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    image = get_file_content('static/images/test.jpg')

    """ 调用植物识别 """

    """ 如果有可选参数 """

    options = {}
    #options["top_num"] = 3
    #options["filter_threshold"] = "0.7"
    options["baike_num"] = 5

    """ 带参数调用植物识别 """
    result=client.plantDetect(image, options)

    print(json.dumps(result))
    return result #返回的是字典对象