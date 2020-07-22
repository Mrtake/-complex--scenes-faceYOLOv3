"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
from keras import backend as k
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"  #这里我们要好好看一下，看那块显卡在用，再进行设置



#同时可能出现显存不足的情况，将batch_size改小就可以了
# conf = k.tf.ConfigProto(devic_count = {"CPU":2},
#                         intra_op_parallenlism_threads=2,
#                         inter_op_parallenlism_threads =2)
# k.set_session(k.tf.Session(config=conf))


def _main():
    annotation_path = 'WIDER_train.txt'
    val_path = 'WIDER_val.txt'  # 验证数据路径
    log_dir = 'logs/000/'
    classes_path = 'model_data/WIDER_classes.txt'
    anchors_path = 'model_data/WIDER_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)  # multiple of 32, hw
    model = create_model(input_shape, anchors, num_classes)
    train(model, annotation_path, val_path, input_shape, anchors, num_classes, log_dir=log_dir)


def train(model, annotation_path, val_path, input_shape, anchors, num_classes, log_dir='logs/'):
    model.compile(optimizer=Adam(lr=1e-3,decay=0.0005), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    logging = TensorBoard(log_dir=log_dir)  # TensorBoard可视化工具
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:3f}-val_loss{val_loss:.3f}.h5',
                                 # 每隔3个epoch存储一次（period）
                                 monitor='val_loss', save_weights_only=True, save_best_only=True,
                                 period=90)  # 该回调函数将在每个epoch后保存模型到filepath;monitor：需要监视的值
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                  verbose=1)  # 当评价指标不在提升时，减少学习率;每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)  # 当监测值不再改善时，该回调函数将中止训练

    batch_size = 8
    with open(annotation_path) as f :
        lines = f.readlines() #读取整个文件，列表形式返回全文，每行作为一个字符串作为列表元素
    np.random.seed(10101)  #产生随机数，里面的数字只决定了随机数的起始位置
    np.random.shuffle(lines) #进行一维随机打乱，增加训练的多样性
    np.random.seed(None)  #不设置种子
    num_train = len(lines) #得到训练的数据长度

    with open(val_path) as f:
        liness =f.readlines() ##读取整个文件，列表形式返回全文，每行作为一个字符串作为列表元素
    np.random.seed(10101)  # 产生随机数，里面的数字只决定了随机数的起始位置
    np.random.shuffle(liness)  # 进行一维随机打乱，增加训练的多样性
    np.random.seed(None)  # 不设置种子
    num_val = len(liness)  # 得到验证的数据长度
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    model.fit_generator(data_generator_wrap(lines, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrap(liness, batch_size, input_shape, anchors,
                                                            num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs= 180,
                        initial_epoch=0,
                        callbacks = [logging, checkpoint, reduce_lr])
    model.save_weights(log_dir + 'trained_weights.h5')


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=False,
                 weights_path='model_data/yolo_weights.h5'):
    K.clear_session()  # get a new session
    h, w = input_shape
    image_input = Input(shape=(None, None, 3))
    num_anchors = len(anchors)
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))


    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers) - 7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()






