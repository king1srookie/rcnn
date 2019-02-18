import os
import codecs
import Networks
import numpy as np
import process_data
import config as cfg
import tensorflow as tf
from sklearn.externals import joblib
slim = tf.contrib.slim
flower = {1:'japanese_chin', 2:'Bombay Cat'}
class Solver :

    def __init__(self, net, data, is_training=False, is_fineturn=False, is_Reg=False):
        self.net = net
        self.data = data
        self.is_Reg = is_Reg
        self.is_fineturn = is_fineturn
        self.summary_step = cfg.Summary_iter#50
        self.save_step = cfg.Save_iter#5000
        self.max_iter = cfg.Max_iter#20000
        self.staircase = cfg.Staircase#true

        if is_fineturn:
            self.initial_learning_rate = cfg.F_learning_rate
            self.decay_step = cfg.F_decay_iter
            self.decay_rate = cfg.F_decay_rate
            self.weights_file = cfg.T_weights_file#r'./output/train_alexnet/save.ckpt-40000'
            self.output_dir = r'./output/fineturn'
        elif is_Reg:
            self.initial_learning_rate = cfg.R_learning_rate
            self.decay_step = cfg.R_decay_iter
            self.decay_rate = cfg.R_decay_rate
            if is_training == True:
                self.weights_file = None
            else:
                self.weights_file = cfg.R_weights_file#r'./output/Reg_box/save.ckpt-10000'
            self.output_dir = r'./output/Reg_box'
        else:
            self.initial_learning_rate = cfg.T_learning_rate#0.0001
            self.decay_step = cfg.T_decay_iter#100
            self.decay_rate = cfg.T_decay_rate#0.99
            if is_training == True:
                self.weights_file = None
            else:
                self.weights_file = cfg.F_weights_file#r'./output/fineturn/save.ckpt-54000'
            self.output_dir = r'./output/train_alexnet'
        self.save_cfg()#保存参数
        #在恢复模型及其参数时，名字的R-CNN/fc_11网络层的参数不进行载入
        exclude = ['R-CNN/fc_11']
        self.variable_to_restore = slim.get_variables_to_restore(exclude=exclude)
        self.variable_to_save = slim.get_variables_to_restore(exclude=[])
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=1)#只想保存最后一代的模型，则只需要将max_to_keep设置为1
        self.saver = tf.train.Saver(self.variable_to_save, max_to_keep=1)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')

        self.summary_op = tf.summary.merge_all()#可以将所有summary全部保存到磁盘
        self.writer = tf.summary.FileWriter(self.output_dir)#指定一个文件用来保存图

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable = False)#定义tensor的维度
        self.learning_rate = tf.train.exponential_decay(
                                                         self.initial_learning_rate,
                                                         self.global_step,
                                                         self.decay_step,
                                                         self.decay_rate,
                                                         self.staircase,
                                                         name='learning_rate'
                                                        )
        #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
        if is_training :
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                                                            self.net.total_loss ,global_step=self.global_step
                                                    )
            self.ema = tf.train.ExponentialMovingAverage(0.99)#移动平均法
            self.average_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([self.optimizer]):#指定某些操作执行的依赖关系
                self.train_op = tf.group(self.average_op)#组合多个操作

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            self.restorer.restore(self.sess, self.weights_file)
        self.writer.add_graph(self.sess.graph)

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__#这个属性中存放着类的属性和方法对应的键值对
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():##说明：检测字符串中所有的字母是否都为大写
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def train(self):
        for step in range(1, self.max_iter+1):#20000
            if self.is_Reg:
                input, labels = self.data.get_Reg_batch()
            elif self.is_fineturn:
                input, labels = self.data.get_fineturn_batch()
            else:
                input, labels = self.data.get_batch()

            feed_dict = {self.net.input_data:input, self.net.label:labels}
            if step % self.summary_step == 0 :#50
                summary, loss, _=self.sess.run([self.summary_op,self.net.total_loss,self.train_op], feed_dict=feed_dict)
                self.writer.add_summary(summary, step)
                print("Data_epoch:"+str(self.data.epoch)+" "*5+"training_step:"+str(step)+" "*5+ "batch_loss:"+str(loss))
            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)
            if step % self.save_step == 0 :#5000
                print("saving the model into " + self.ckpt_file)
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)

    def predict(self, input_data):
        feed_dict = {self.net.input_data :input_data}
        predict_result = self.sess.run(self.net.logits, feed_dict = feed_dict)
        return predict_result

def get_Solvers():
    '''
    此函数用于得到三个Solver：特征提取的Solver，SVM预测分类的solver，Reg_Box预测框回归的Solver
    
    :return: 
    '''
    weight_outputs = ['train_alexnet', 'fineturn', 'SVM_model', 'Reg_box']
    for weight_output in weight_outputs:
        output_path = os.path.join(cfg.Out_put, weight_output)#'./output'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    if len(os.listdir(r'./output/train_alexnet')) == 0:
        Train_alexnet = tf.Graph()
        with Train_alexnet.as_default():
            Train_alexnet_data = process_data.Train_Alexnet_Data()
            Train_alexnet_net = Networks.Alexnet_Net(is_training=True, is_fineturn=False, is_SVM=False)
            Train_alexnet_solver = Solver(Train_alexnet_net, Train_alexnet_data, is_training=True, is_fineturn=False, is_Reg=False)
            Train_alexnet_solver.train()

    if len(os.listdir(r'./output/fineturn')) == 0:
        Fineturn = tf.Graph()
        with Fineturn.as_default():
            Fineturn_data = process_data.FineTun_And_Predict_Data()
            Fineturn_net = Networks.Alexnet_Net(is_training=True, is_fineturn=True, is_SVM=False)
            Fineturn_solver = Solver(Fineturn_net, Fineturn_data, is_training=True, is_fineturn=True, is_Reg=False)
            Fineturn_solver.train()

    Features = tf.Graph()
    with Features.as_default():
        Features_net = Networks.Alexnet_Net(is_training=False, is_fineturn=True, is_SVM=True)
        Features_solver = Solver(Features_net, None, is_training=False, is_fineturn=True, is_Reg=False)
        Features_data = process_data.FineTun_And_Predict_Data(Features_solver, is_svm=True, is_save=True)

    svms = []
    if len(os.listdir(r'./output/SVM_model')) == 0:
        SVM_net = Networks.SVM(Features_data)
        SVM_net.train()
    for file in os.listdir(r'./output/SVM_model'):
        svms.append(joblib.load(os.path.join('./output/SVM_model', file)))

    Reg_box = tf.Graph()
    with Reg_box.as_default():
        Reg_box_data = Features_data
        Reg_box_net = Networks.Reg_Net(is_training=True)
        if len(os.listdir(r'./output/Reg_box')) == 0:
            Reg_box_solver = Solver(Reg_box_net, Reg_box_data, is_training=True, is_fineturn=False, is_Reg=True)
            Reg_box_solver.train()
        else:
            Reg_box_solver = Solver(Reg_box_net, Reg_box_data, is_training=False, is_fineturn=False, is_Reg=True)

    return Features_solver, svms, Reg_box_solver

if __name__ =='__main__':
    
    Features_solver, svms, Reg_box_solver =get_Solvers()

    img_path = './2Pets/jpg/0/image_0561.jpg'
    imgs, verts = process_data.image_proposal(img_path)
    process_data.show_rect(img_path, verts, ' ')
    features = Features_solver.predict(imgs)
    print(np.shape(features))

    results = []
    results_old = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            # not background
            if pred[0] != 0:
                results_old.append(verts[count])
                #print(Reg_box_solver.predict([f.tolist()]))
                if Reg_box_solver.predict([f.tolist()])[0][0] > 0.1:#0.5
                    px, py, pw, ph = verts[count][0], verts[count][1], verts[count][2], verts[count][3]
                    old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0
                    x_ping, y_ping, w_suo, h_suo = Reg_box_solver.predict([f.tolist()])[0][1], \
                                                   Reg_box_solver.predict([f.tolist()])[0][2], \
                                                   Reg_box_solver.predict([f.tolist()])[0][3], \
                                                   Reg_box_solver.predict([f.tolist()])[0][4]
                    new__center_x = x_ping * pw + old_center_x
                    new__center_y = y_ping * ph + old_center_y
                    new_w = pw * np.exp(w_suo)
                    new_h = ph * np.exp(h_suo)
                    new_verts = [new__center_x, new__center_y, new_w, new_h]
                    results.append(new_verts)
                    results_label.append(pred[0])
        count += 1

    average_center_x, average_center_y, average_w,average_h = 0, 0, 0, 0
    #给预测出的所有的预测框区一个平均值，代表其预测出的最终位置
    print('results')
    print(results)
    for vert in results:
        average_center_x += vert[0]
        average_center_y += vert[1]
        average_w += vert[2]
        average_h += vert[3]
    average_center_x = average_center_x / len(results)
    average_center_y = average_center_y / len(results)
    average_w = average_w / len(results)
    average_h = average_h / len(results)
    average_result = [[average_center_x, average_center_y, average_w, average_h]]
    result_label = max(results_label, key=results_label.count)
    process_data.show_rect(img_path, results_old,' ')
    process_data.show_rect(img_path, average_result,flower[result_label])




   









