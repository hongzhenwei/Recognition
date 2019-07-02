#导入相关的软件包

from werkzeug.utils import secure_filename
import os #文件路径
import cv2 #图片处理

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
#判断文件名是否合法函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def upload(request):
    if request.method == 'POST':#成立 上传图片文件

        f = request.files['file'] #上传图片文件对象

        if not (f and allowed_file(f.filename)):
            #不满足要求的图片文件,生成出错的json字符串
            #return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
            return 2
        #获取心情的字符串
        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path) #保存上传的图片文件

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        # return render_template('dish.html', userinput=user_input, val1=time.time())
        return 1
    #处理GET请求的
    # return render_template('dish.html')
    return 0