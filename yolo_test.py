
import colorsys
import os
from timeit import default_timer as timer
import time #外部计时

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

path = 'C:/Users/hp/Desktop/123' #需要检测的图片路径，这里是批量的图片,路径自己设置
# 创建创建一个存储检测结果的dir
result_path = './result'
if not os.path.exists(result_path): #如果不存在
    os.makedirs(result_path) #创建目录，只创建一次

# result如果之前存放的有文件，全部清除
for i in os.listdir(result_path):# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    path_file = os.path.join(result_path,i)
    if os.path.isfile(path_file): #用于判断某一对象(需提供绝对路径)是否为文件
        os.remove(path_file) #是就全部去掉

#创建一个记录检测结果的文件
txt_path =result_path + '/result.txt'
file = open(txt_path,'w')

class YOLO(object): #YOLO类的初始化参数，这里创建了一个类
    _defaults = {
        "model_path": "logs/000/trained_weights.h5", #已经训练好的模型,自定义训练的检测，人脸或者其他
        "anchors_path": "model_data/WIDER_anchors.txt", #通过聚类算法得到的3组9个anchor box，分别用于13x13、26x26、52x52的feature map
        "classes_path": "model_data/WIDER_classes.txt", #自己定义的类
        "score": 0.3, #框置信度阈值阈值，小于阈值则目标框被抛弃
        "iou": 0.45, #IOU（Intersection over Union）阈值，大于阈值的重叠框被删除
        "model_image_size":(416, 416), #输入图片的“标准尺寸”，不同于这个尺寸的输入会首先调整到标准大小
        "gpu_num": 1,#GPU数量，通常是指Nvidia的GPU
    } #总体而言，这是一个词典  #定义类的基本属性

    @classmethod # 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n +"'"

    def __init__(self, **kwargs): #**kwargs是词典格式 #定义构造方法
        self.__dict__.update(self._defaults) #__dict__是内置成员词典，通过_defaults词典更新内置词典，得到一个新的词典
        self.__dict__.update(kwargs) #此时的__dict__内置词典已经更新，此时再更新后传入kwargs中，构成方法的参数
        self.class_names = self._get_class()  #读取的形式为['aeroplane\n', 'bicycle\n', 'bird\n', 'boat\n',.....]
        self.anchors = self._get_anchors() #将txt文件的框架其变为2维数组形式,列是2
        self.sess = K.get_session() #首先我们需要创建一个Session对象
        self.boxes, self.scores, self.classes = self.generate()



    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path) #把path中包含的"~"和"~user"转换成用户目录,针对不同系统的
        with open(classes_path) as f: #打开txt文件夹
            class_name = f.readlines() #一次性全部读取文件
        class_names = [c.strip() for c in class_name] #一个个读取出来以列表形式存储
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path) #把path中包含的"~"和"~user"转换成用户目录,针对不同系统的
        with open(anchors_path) as f: #打开txt文件
            anchors = f.readline() #一行一行的读取文件
        anchors = [ float(x) for x in anchors.split(',')] #去掉',"这个字符
        return np.array(anchors).reshape(-1, 2) #将其变为数组，将txt文件的框架其变为2维数组形式,列是2

    def generate(self): #读取：model路径、anchor box、coco类别，加载模型yolo_weights.h5
        model_path = os.path.expanduser(self.model_path) #把path中包含的"~"和"~user"转换成用户目录,针对不同系统的
        assert model_path.endswith('.h5') #判断文件格式是不是.h5结尾。采用断言语句

        num_anchors = len(self.anchors) #先验框长度
        num_classes = len(self.class_names) #类别长度
        is_tiny_version = num_anchors==6 #若采用小型网络，只有2个输出，所以才有6个框架，判断是否使用小网络
        try: #try/except语句用来检测try语句块中的错误，从而让except语句捕获异常信息并处理。
            self.yolo_model = load_model(model_path, compile=False)  #运行别的代码,加载模型(读取网络和权重)
        except:  #这行语句的是利用给定框架来判断使用哪个网络,当try语句有异常是执行，若没有异常不执行
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None,3)),num_anchors//2, num_classes)\
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)),num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:  #如果没有异常发生,这里是看模型输出层的最后一层的最后一个维度。yolo_model.output=3
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                '模型与给定的锚点和类大小之间不匹配'

        print('{} model, anchors, and classes loaded.'.formate(model_path)) #一行语句，表示模型，框架全部加载

        # 对于80种coco目标，确定每一种目标框的绘制颜色，即：将(x/80, 1.0, 1.0)的颜色转换为RGB格式，并随机调整颜色以便于肉眼识别，其中：一个1.0表示饱和度，一个1.0表示亮
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                for x in range(len(self.class_names))] #[(0.0, 1.0, 1.0), (0.0125, 1.0, 1.0), (0.025, 1.0, 1.0), ...(0.9875, 1.0, 1.0)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)) #hsv转换为rgb[(1.0, 0.0, 0.0), (1.0, 0.07499999999999996, 0.0), (1.0, 0.15000000000000002, 0.0),...(1.0, 0.0, 0.07499999999999929)]
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors)) #[(255, 0, 0), (255, 19, 0), (255, 38, 0), (255, 57, 0),....(255, 0, 57), (255, 0, 38), (255, 0, 19)]
        np.random.seed(10101)  #固定种子，可在每次运行中获得一致的颜色。
        np.random.shuffle(self.colors) #随机排列颜色以关联相邻的类。
        np.random.seed(None) #将种子重置为默认值。

        #为过滤后的边界框生成输出张量目标。
        self.input_image_shape = K.placeholder(shape=(2,)) #keras中的占位符,此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        if self.gup_num>=2: #若GPU个数大于等于2，调用multi_gpu_model(),并行运行模型
            self.yolo_model = multi_gpu_model(self.yolo_model, gups=self.gpu_num)  #model: Keras模型对象,gpus: 大或等于2的整数，要并行的GPU数目
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes #这个函数等会会在后面调用，喂入数据

    def detect_image(self, image): #定义图像检测函数
        start = timer() #计时开始函数

        if self.model_image_size != (None, None): #判断图片是否存在,如果存在接着执行下面语句,# 416x416, 416=32*13可以是其他尺寸，只要满足条件
            assert self.model_image_size[0]%32 == 0 ,'Multiples of 32 required' #图像的w,必须是32的整数倍,如果不是触发异常
            assert self.model_image_size[1]%32 ==0 ,'Multiples of 32 required'  #图像的h,必须是32的整数倍,如果不是触发异常
            # 调用letterbox_image()函数，即：先生成一个用“绝对灰”R128-G128-B128“填充的416x416新图片，然后用按比例缩放（采样方法：BICUBIC）后的输入图片粘贴，粘贴不到的部分保留为灰色
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size))) #后面的语句实际上是元组，也就是尺寸大小,新的图像出现
        else: #if后面的语句不成立
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32)) #新的尺寸元组
            boxed_image = letterbox_image(image, new_image_size) #填充后新的图像
        image_data = np.array(boxed_image, dtype='float') #图像转换为数组，同时规定其类型

        print(image_data.shape) #输出其图片的尺寸
        image_data /= 255. #图片归一化
        image_data = np.expand_dims(image_data, 0) ##批量添加一维 -> (1,416,416,3) 为了符合网络的输入格式 -> (bitch, w, h, c)
         #运行次回话
        out_boxes, out_scores, out_classes = self.sess.run(  #目的为了求boxes,scores,classes，具体计算方式定义在generate（）函数内。
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data, #图像数据
                self.input_image_shape: [image.size[1], image.size[0]], #图像尺寸,输入到了generate函数接口里面
                K.learning_phase(): 0  #学习模式 0：测试模型。 1：训练模式
             })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 加载字体创建字体对象
        thickness = (image.size[0] + image.size[1]) // 300  # 设置目标框线条的宽度

        file.write(str(len(out_boxes))+';')

        for i, c in reversed(list(enumerate(out_classes))): #反向元组迭代;#对于c个目标类别中的每个目标框i，调用Pillow画图
            predicted_class = self.class_names[c] #输出类型
            box = out_boxes[i] #此类型所属框架
            score = out_scores[i] #此框架得分

            label ='{} {:.2f}'.formate(predicted_class, score) ##标签;两个都为保留小数后两位
            draw = ImageDraw.Draw(image) #输出：绘制输入的原始图片 创建一个可用来Image操作的对象
            label_size = draw.textsize(label, font) #返回label的宽和高（多少个pixels）,给定字符串的大小

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32')) #目标框的上、左两个坐标小数点后一位四舍五入，上边的y
            left = max(0, np.floor(left + 0.5).astype('int32')) #左边x
            bottom = min(image[1], np.floor(bottom + 0.5).astype('int32')) #目标框的下、右两个坐标小数点后一位四舍五入，与图片的尺寸相比，取最小值；下面的y
            right = min(image.size[0], np.floor(right + 0.5).astype('int32')) #右边的x

            file.write(str(left)+' '+str(top)+' '+str(right-left)+' '+str(bottom-top)+' '+str(score)+';')#人脸检测必须

            print(label, (left, top), (right - left, bottom - top))

            if top - label_size[1] >=0: #确定标签（label）框起始点位置：左、下
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness): #画目标框，线条宽度为thickness
                draw.rectangle(
                    [left + i, top + i, right - i, bottom -i],
                    outline=self.colors[c]) #绘制一个长边形。
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])  #画标签框
            draw.text(text_origin, label, fill=(0, 0, 0), font=font) #填写标签内容
            del draw

        end = timer()
        print('time consume:%.3f s '%(end - start))
        return image

    def close_session(self):  # 关闭回话
        self.sess.close()


# 图片检测

if __name__=='__main__':

    t1 = time.time()
    yolo = YOLO()
    for filename in os.listdir(path): #文件夹包含的文件或文件夹的名字的列表
        image_path = path + '/' + filename #每个图片的路径
        portion = os.path.split(image_path) #os.path.split()：按照路径将文件名和路径分割开
        file.write(portion[1] + ';') #写入文件名
        image = Image.open(image_path) #打开图片
        r_image = yolo.detect_image(image)  #进行检测
        file.write('\n')
        # r_image.show() 显示检测结果，不需要看，我只想要结果
        image_save_path = './result/result_' + portion[1] #保持的路径，前面要不要加result，其实随便
        print('detect result save to....:' + image_save_path)
        r_image.save(image_save_path) #将文件保存

    time_sum = time.time() - t1 #计算总时间
    # file.write('time sum: '+str(time_sum)+'s')，要不要写到文件里面待定
    print('time sum:', time_sum)
    file.close()
    yolo.close_session()