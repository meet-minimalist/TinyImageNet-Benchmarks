# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:19:33 2019

@author: Meet
"""

import os
import numpy as np
from tqdm import tqdm
import datetime, shutil, glob, cv2

import config
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tfrecords_helper import get_batch

class Classifier:
    def __init__(self, model_name):
        tf.reset_default_graph()
        self.input_dims = config.input_dims
        self.num_classes = config.num_classes
        self.model_name = model_name

    def model(self, x, is_training):
        logits, output, self.minLR, self.maxLR, self.step_factor, self.weight_decay = config.get_model(self.model_name, x, is_training)
        return logits, output

    def get_loss_and_accuracy(self, label, logits, output):
        loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
        loss_reg = tf.losses.get_regularization_loss()
        
        """
        loss_reg = 0
        for var in tf.global_variables():
            if 'kernel' in var.name:
                loss_reg += self.weight_decay * tf.nn.l2_loss(var)
        """
        #loss_reg = tf.constant(0.0, dtype=tf.float32)
        total_loss = loss_cls + loss_reg
        
        top_1_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=-1), tf.argmax(label, axis=-1)), tf.float32)) * 100
        top_5_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(output, tf.argmax(label, axis=-1), 5), tf.float32)) * 100
        top_1_correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, axis=-1), tf.argmax(label, axis=-1)), tf.float32))
        top_5_correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(output, tf.argmax(label, axis=-1), 5), tf.float32))
        return total_loss, loss_cls, loss_reg, top_1_accuracy, top_5_accuracy, top_1_correct, top_5_correct
    
    def get_clr(self, g_step):
        #init_lr = 0.01
        #k = 1
        #steps_per_epoch = int(np.ceil(config.train_img_cnt / config.batch_size))
        #lr = init_lr * np.exp(-k * g_step / steps_per_epoch)
        
        cycle = np.floor(1 + g_step / (2 * self.step_size))
        x = np.abs(g_step / self.step_size - 2 * cycle + 1)
        lr = self.minLR + (self.maxLR - self.minLR) * np.maximum(0, (1-x))
        return lr
        
    def custom_summary(self, sum_writer, global_step_value, mode=None, l_total=None, l_cls=None, l_reg=None, top_1_acc=None, top_5_acc=None, image=None, lr=None):
        # mode can be 'train', 'test', 'valid'
        sum_list = []
        value_list = [l_total, l_cls, l_reg, top_1_acc, top_5_acc, image, lr]
        name = ['l_total', 'l_cls', 'l_reg', 'top_1_acc', 'top_5_acc', 'image', 'lr']
        for c, value in enumerate(value_list):                            
            if value is not None:
                if c == 5:
                    retval, buffer = cv2.imencode('.jpg', image)
                    img_sum = tf.Summary.Image(encoded_image_string=buffer.tostring(),
                                                       height=image.shape[0],
                                                       width=image.shape[1])
                    if mode == None:
                        summ_ = tf.Summary.Value(tag=name[c], image=img_sum)
                    else:
                        summ_ = tf.Summary.Value(tag=mode + '/' + name[c], image=img_sum)
                else:
                    if mode == None:
                        summ_ = tf.Summary.Value(tag=name[c], simple_value=value_list[c])
                    else:
                        summ_ = tf.Summary.Value(tag=mode + '/' + name[c], simple_value=value_list[c])
                sum_list.append(summ_)
                
        summary = tf.Summary(value=sum_list)
        sum_writer.add_summary(summary, global_step_value)
        sum_writer.flush()

    def dry_run_clr(self):
        time = str(datetime.datetime.now())
        time = time.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")
        path = config.summary_path + "/" + self.model_name
        os.makedirs(path, exist_ok=True)
        summaries_path = config.summary_path + "/" + self.model_name + "/" + time + "_dry_run_clr"
        os.makedirs(summaries_path, exist_ok=True)
        
        current_files = glob.glob("*")
        for i in range(len(current_files)):
            if os.path.isfile(current_files[i]):
                shutil.copy2(current_files[i], summaries_path)
        

        model_files = glob.glob("./models/*")
        os.makedirs(summaries_path + "/models", exist_ok=True)
        for model_file in model_files:
            if self.model_name in model_file.lower():
                shutil.copy2(model_file, summaries_path + "/models/")
        
        x = tf.placeholder(shape=[None, self.input_dims[0], self.input_dims[1], 3], dtype=tf.float32, name='input')
        label = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32, name='label')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        
        logits, output = self.model(x, is_training)
        
        total_loss, _, _, _, _, _, _ = self.get_loss_and_accuracy(label, logits, output)
        
        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name='global_step')
        lr = tf.train.exponential_decay(config.start_lr, global_step, config.decay_steps, config.decay_rate)
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(total_loss, global_step)

        train_initializer, img_train_data, label_train_data = get_batch(config.train_tfrecord_list, config.batch_size, augment=True)

        ### Restrict the GPU usage for training if possible. ###
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(summaries_path)
            #writer_train.add_graph(sess.graph)
            for i in range(config.total_steps):                
                try:
                    imgs, labels = sess.run([img_train_data, label_train_data])
                    _, l = sess.run([opt, total_loss], feed_dict={x: imgs, label: labels, is_training: True})
                    
                    self.custom_summary(writer_train, tf.train.global_step(sess, global_step), None, l, None, None, None, None, None, sess.run(lr))                    
                    
                    print("Step: {}, Loss: {:.2f}".format(tf.train.global_step(sess, global_step), l))

                except:
                    sess.run(train_initializer.initializer)
            print("Iteration Completed.")

    def plot_clr_graph(self):
        time = str(datetime.datetime.now())
        time = time.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")
        path = config.summary_path + "/" + self.model_name
        os.makedirs(path, exist_ok=True)
        summaries_path = config.summary_path + "/" + self.model_name + "/" + time + "_clr_plot/"
        os.makedirs(summaries_path, exist_ok=True)

        steps_per_epoch = int(np.ceil(config.train_img_cnt / config.batch_size))
        self.step_size = self.step_factor * steps_per_epoch

        writer = tf.summary.FileWriter(summaries_path)

        step_counter = 0
        for eps in tqdm(range(config.epochs)):
            for batch in range(steps_per_epoch):
                if (step_counter + 1) % (2 * self.step_size) == 0:
                    if self.maxLR > self.minLR:
                        multiplier = (step_counter + 1) // (2 * self.step_size)
                        self.maxLR = (self.maxLR) * (0.99**multiplier)
                    else:
                        self.maxLR = self.minLR
                lr = self.get_clr(step_counter)
                if (batch + 1) % 10 == 0:
                    self.custom_summary(writer, step_counter, lr=lr)
                step_counter += 1
				
    def train(self, resume=False, resume_from_eps=0, resume_from_gstep=0, restore_ckpt=None):
        ### Preprocessing stuff ###
        time = str(datetime.datetime.now())
        time = time.replace(":", "_").replace(" ", "_").replace("-", "_").replace(".", "_")
        path = config.summary_path + "/" + self.model_name
        os.makedirs(path, exist_ok=True)
        summaries_path = config.summary_path + "/" + self.model_name + "/" + time + "_training_summary"
        os.makedirs(summaries_path, exist_ok=True)
        ckpt_path = summaries_path + "/best_checkpoint/"
        os.makedirs(ckpt_path, exist_ok=True)
		
        current_files = glob.glob("*")
        for i in range(len(current_files)):
            if os.path.isfile(current_files[i]):
                shutil.copy2(current_files[i], summaries_path)
        
        model_files = glob.glob("./models/*")
        os.makedirs(summaries_path + "/models", exist_ok=True)
        for model_file in model_files:
            if self.model_name in model_file.lower():
                shutil.copy2(model_file, summaries_path + "/models/")
        
        ### Placeholder definitions ###
        x = tf.placeholder(shape=[None, self.input_dims[0], self.input_dims[1], 3], dtype=tf.float32, name='input')
        label = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32, name='label')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        lr = tf.placeholder(dtype=tf.float32, name='learning_rate')
        
        ### Model definitions ###
        logits, output = self.model(x, is_training)
        
        ### Loss and accuracy definitions ###
        total_loss, loss_cls, loss_reg, top_1_accuracy, top_5_accuracy, top_1_correct, top_5_correct = self.get_loss_and_accuracy(label, logits, output)
        
        ### Optimizer definitions ###
        global_step = tf.Variable(initial_value=resume_from_gstep, dtype=tf.int32, trainable=False, name='global_step')
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            opt = tf.train.AdamOptimizer(lr, beta1=0.9).minimize(total_loss, global_step)
                 
        ### TF Records reading ###
        train_initializer, img_train_data, label_train_data = get_batch(config.train_tfrecord_list, config.batch_size, augment=True)
        img_val_data, label_val_data = get_batch(config.test_tfrecord_list, config.batch_size, augment=False, is_validation_set=True)
        test_initializer, img_test_data, label_test_data = get_batch(config.test_tfrecord_list, config.batch_size, augment=False)

        ### For Learning rate scheduling ###
        steps_per_epoch = int(np.ceil(config.train_img_cnt / config.batch_size))
        self.step_size = self.step_factor * steps_per_epoch
        
        ### Checkpoint Saving mechanism ###
        saver = tf.train.Saver()
        restorer = tf.train.Saver() 
        # restore all variables

        ### Best accuracy is set to 0.0 before starting training. ###
        best_acc = 0
        
        ### Restrict the GPU usage for training if possible. ###
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
               
        ### Start training. ###
        with tf.Session(config=sess_config) as sess:
            if resume:
                restorer.restore(sess, restore_ckpt)
            else:
                sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter(summaries_path + "/train")
            writer_test = tf.summary.FileWriter(summaries_path + "/test")
            writer_valid = tf.summary.FileWriter(summaries_path + "/valid")
            writer_train.add_graph(sess.graph)
            
            for eps in range(resume_from_eps, config.epochs):
                file = open(summaries_path + "/log.txt", 'a')
                sess.run(train_initializer.initializer)
                
                for batch in range(steps_per_epoch):
                    if (tf.train.global_step(sess, global_step) + 1) % (2 * self.step_size) == 0:
                        if self.maxLR > self.minLR:
                            multiplier = (tf.train.global_step(sess, global_step) + 1) // (2 * self.step_size)
                            self.maxLR = (self.maxLR) * (0.99**multiplier)
                        else:
                            self.maxLR = self.minLR

                    imgs_tr, label_tr = sess.run([img_train_data, label_train_data])
                    _, l, l_cls, l_reg, b_acc_t_1, b_acc_t_5 = sess.run([opt, total_loss, loss_cls, loss_reg, top_1_accuracy, top_5_accuracy], feed_dict={x: imgs_tr, label: label_tr, is_training: True, lr: self.get_clr(tf.train.global_step(sess, global_step))})
                    
                    self.custom_summary(writer_train, tf.train.global_step(sess, global_step), None, l, l_cls, l_reg, b_acc_t_1, b_acc_t_5, image=None, lr=self.get_clr(tf.train.global_step(sess, global_step)))
                    
                    print("Epoch: {}/{}, Batch No.: {}/{}, Total Loss: {:.2f}, Loss Cls: {:.2f}, Loss Reg: {:.2f}, Top-1 Accuracy: {:.2f}, Top-5 Accuracy: {:.2f}".format(eps+1, config.epochs, batch+1, steps_per_epoch, l, l_cls, l_reg, b_acc_t_1, b_acc_t_5))
                    file.write("Epoch: {}/{}, Batch No.: {}/{}, Total Loss: {:.2f}, Loss Cls: {:.2f}, Loss Reg: {:.2f}, Top-1 Accuracy: {:.2f}, Top-5 Accuracy: {:.2f}\n".format(eps+1, config.epochs, batch+1, steps_per_epoch, l, l_cls, l_reg, b_acc_t_1, b_acc_t_5))
                    
                    if (batch + 1) % 100 == 0:
                        # get loss on 10 validation batches
                        val_l_total = []
                        val_l_cls_total = []
                        val_l_reg_total = []
                        val_acc_t_1 = []
                        val_acc_t_5 = []
                        for _ in range(10):
                            imgs_val, label_val= sess.run([img_val_data, label_val_data])
                            val_l, val_l_cls, val_l_reg, v_acc_t_1, v_acc_t_5 = sess.run([total_loss, loss_cls, loss_reg, top_1_accuracy, top_5_accuracy], feed_dict={x: imgs_val, label: label_val, is_training: False})
                            val_l_total.append(val_l)
                            val_l_cls_total.append(val_l_cls)
                            val_l_reg_total.append(val_l_reg)
                            val_acc_t_1.append(v_acc_t_1)
                            val_acc_t_5.append(v_acc_t_5)
                        
                        self.custom_summary(writer_valid, tf.train.global_step(sess, global_step), None, np.mean(val_l_total), np.mean(val_l_cls_total), np.mean(val_l_reg_total), np.mean(val_acc_t_1), np.mean(val_acc_t_5), None, None)
                        print("Epoch: {}/{} completed, Last Train Loss: {:.2f}, Valid Loss: {:.2f}, Accuracy: {:.2f}".format(eps+1, config.epochs, l, np.mean(val_l_total), np.mean(val_acc_t_1), np.mean(val_acc_t_5)))
                        file.write("Epoch: {}/{} completed, Last Train Loss: {:.2f}, Valid Loss: {:.2f}, Accuracy: {:.2f}\n".format(eps+1, config.epochs, l, np.mean(val_l_total), np.mean(val_acc_t_1), np.mean(val_acc_t_5)))
                        
                        for ctr in range(8):  ### Total 4 images will be displayed as batch_size is 4
                            idx = np.random.choice(imgs_tr.shape[0], 1)[0]
                            single_img = np.uint8(imgs_tr[idx] * 255)
                            self.custom_summary(writer_train, tf.train.global_step(sess, global_step), 'image_train_' + str(ctr), None, None, None, None, None, single_img, None)

                test_l_total = []
                test_l_cls_total = []
                test_l_reg_total = []
                test_acc_t_1 = 0
                test_acc_t_5 = 0
                sess.run(test_initializer.initializer)
                for _ in tqdm(range(int(np.ceil(config.test_img_cnt / config.batch_size)))):
                    imgs_test, label_test = sess.run([img_test_data, label_test_data])
                    test_l, test_l_cls, test_l_reg, t_cnt_t_1, t_cnt_t_5 = sess.run([total_loss, loss_cls, loss_reg, top_1_correct, top_5_correct], feed_dict={x: imgs_test, label: label_test, is_training: False})
                    test_l_total.append(test_l)
                    test_l_cls_total.append(test_l_cls)
                    test_l_reg_total.append(test_l_reg)
                    test_acc_t_1 += t_cnt_t_1
                    test_acc_t_5 += t_cnt_t_5
                
                test_acc_t_1 = (test_acc_t_1 * 100 / config.test_img_cnt)
                test_acc_t_5 = (test_acc_t_5 * 100 / config.test_img_cnt)

                self.custom_summary(writer_test, tf.train.global_step(sess, global_step), None, np.mean(test_l_total), np.mean(test_l_cls_total), np.mean(test_l_reg_total), np.mean(test_acc_t_1), np.mean(test_acc_t_5), None, None)
                print("Epoch: {}/{}, Test Loss: {:.2f}, Top-1 Accuaracy: {:.2f}, Top-5 Accuracy: {:.2f}".format(eps+1, config.epochs, np.mean(test_l_total), np.mean(test_acc_t_1), np.mean(test_acc_t_5)))
                file.write("Epoch: {}/{}, Test Loss: {:.2f}, Top-1 Accuaracy: {:.2f}, Top-5 Accuracy: {:.2f}\n".format(eps+1, config.epochs, np.mean(test_l_total), np.mean(test_acc_t_1), np.mean(test_acc_t_5)))

                if np.mean(test_acc_t_1) > best_acc:                    
                    save_path = ckpt_path + self.model_name + "_eps{}-test_loss_{:.2f}-test_top_1_acc_{:.2f}.ckpt".format(eps+1, np.mean(test_l_total), np.mean(test_acc_t_1))
                    saver.save(sess, save_path)
                    print("Best Checkpoint saved.")                        
                    file.write("Best Checkpoint saved.\n")
                    best_acc = np.mean(test_acc_t_1)
                print("Epoch {} completed.".format(eps+1))                        
                file.write("Epoch {} completed.\n".format(eps+1))
                file.close()
            file = open(summaries_path + "/log.txt", 'a')
            print("Training Completed.")
            file.write("Training Completed.\n")
            file.close()
            
    def eval_on_dataset(self, ckpt_path, train_dataset):
        x = tf.placeholder(shape=[None, self.input_dims[0], self.input_dims[1], 3], dtype=tf.float32, name='input')
        label = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32, name='label')
        
        _, output = self.model(x, False)
            
        top_1_correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(output, axis=-1), tf.argmax(label, axis=-1)), tf.float32))
        top_5_correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(output, tf.argmax(label, axis=-1), 5), tf.float32))
        
        ### TF Records reading ###
        if train_dataset:
            data_initializer, img_data, label_data = get_batch(config.train_tfrecord_list, config.batch_size, augment=False, is_validation_set=False)
            steps_per_epoch = int(np.ceil(config.train_img_cnt / config.batch_size))
            total_imgs = config.train_img_cnt
        else:
            data_initializer, img_data, label_data = get_batch(config.test_tfrecord_list, config.batch_size, augment=False, is_validation_set=False)
            steps_per_epoch = int(np.ceil(config.test_img_cnt / config.batch_size))
            total_imgs = config.test_img_cnt

        restorer = tf.train.Saver()                    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            restorer.restore(sess, ckpt_path)
            top_1_acc = 0
            top_5_acc = 0

            sess.run(data_initializer.initializer)

            for _ in tqdm(range(steps_per_epoch)):
                img_ip, label_ip = sess.run([img_data, label_data])
                t_1, t_5 = sess.run([top_1_correct, top_5_correct], feed_dict={x: img_ip, label: label_ip})
                top_1_acc += t_1
                top_5_acc += t_5

            top_1_acc = top_1_acc * 100 / total_imgs
            top_5_acc = top_5_acc * 100 / total_imgs

            print("Top-1 Accuracy on validation set: ", top_1_acc)
            print("Top-5 Accuracy on validation set: ", top_5_acc)
