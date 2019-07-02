import numpy as np
from flask import Flask, jsonify, render_template, request
import baiduapp,time
from datetime import timedelta #时间
import cv2,os

# import testRegression
import testCNN

# webapp
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/detail1')
def main1():
    return render_template('detail1.html')

@app.route('/detail2')
def main2():
    return render_template('detail2.html')

@app.route('/index')
def main3():
    return render_template('index.html')

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
#判断文件名是否合法函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#上传图片
@app.route('/dish', methods=['POST','GET'])
def dish_upload():
    if request.method == 'POST':
        f = request.files['file']  # 上传图片文件的对象
        if not (allowed_file(f.filename)):
            return jsonify({"error": 101, "msg": "请检查上传的图片类型,仅限于 png,jpg,JPG,PNG,bmp"})

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/images', f.filename)
        f.save(upload_path)  # 保存上传的图片
        # 将上传的文件统一转换为jpg格式的test.jpg文件
        #解决opencv读取图片带中文路径会报错的问题
        img = cv2.imdecode(np.fromfile(upload_path,dtype=np.uint8), -1)
        cv2.imencode('.jpg', img)[1].tofile(basepath + '/static/images/test.jpg')
        print(basepath)
        return render_template('dish.html')
    return render_template('dish.html')


#识别图像
@app.route('/dishvis', methods=['POST'])
def dish_vis():
    result=baiduapp.myDishDetect()
    return jsonify(result)#将baidu返回的数据格式化json


#上传图片
@app.route('/animal', methods=['POST','GET'])
def animal_upload():
    if request.method == 'POST':
        f = request.files['file']  # 上传图片文件的对象
        if not (allowed_file(f.filename)):
            return jsonify({"error": 101, "msg": "请检查上传的图片类型,仅限于 png,jpg,JPG,PNG,bmp"})

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/images', f.filename)
        f.save(upload_path)  # 保存上传的图片
        # 将上传的文件统一转换为jpg格式的test.jpg文件
        #解决opencv读取图片带中文路径会报错的问题
        img = cv2.imdecode(np.fromfile(upload_path,dtype=np.uint8), -1)
        cv2.imencode('.jpg', img)[1].tofile(basepath + '/static/images/test.jpg')
        return render_template('animal.html')
    return render_template('animal.html')

#识别图像
@app.route('/animalvis', methods=['POST'])
def animal_vis():
    result=baiduapp.myanimalDetect()
    return jsonify(result)

#上传图片
@app.route('/plant', methods=['POST','GET'])
def plant_upload():
    if request.method == 'POST':
        f = request.files['file']  # 上传图片文件的对象
        if not (allowed_file(f.filename)):
            return jsonify({"error": 101, "msg": "请检查上传的图片类型,仅限于 png,jpg,JPG,PNG,bmp"})

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/images', f.filename)
        f.save(upload_path)  # 保存上传的图片
        # 将上传的文件统一转换为jpg格式的test.jpg文件
        #解决opencv读取图片带中文路径会报错的问题
        img = cv2.imdecode(np.fromfile(upload_path,dtype=np.uint8), -1)
        cv2.imencode('.jpg', img)[1].tofile(basepath + '/static/images/test.jpg')
        return render_template('plant.html')
    return render_template('plant.html')

#识别图像
@app.route('/plantvis', methods=['POST'])
def plant_vis():
    result=baiduapp.myplantDetect()
    return jsonify(result)

@app.route('/api/conv', methods=['post'])
def conv_mnist():
    print("from client")
    print(request.json)
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output2 = testCNN.convolutional(input)
    print("output2:",output2)
    print(jsonify(results=[output2]))
    return jsonify(results=[output2])


@app.route('/conv')
def conv():
    return render_template('conv.html')


if __name__ == '__main__':
    app.debug=True
    app.run(host='127.0.0.1',port=5000)
