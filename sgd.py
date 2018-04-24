# coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from csv import DictReader
from collections import defaultdict

class SGD(object):
	def __init__(self, data_path):
		
		self.DATA_PATH = data_path
		self.COLUMNS = ['size', 'room', 'price'] #默认的数据集共有三列

		self.sgds = ['SGD','Momentum','NAG','Adagrad','Adadelta','RMSprop','Adam']
		self.sgd_type = self.sgds[0] #选定改进型SGD算法

		self.learning_rate = 0.01    #学习率
		self.training_epochs = 1000  #迭代次数
		self.display_step = 100      #迭代多少次显示一次结果

		self.columns = defaultdict(list)
		self.data = pd.read_csv(tf.gfile.Open(self.DATA_PATH), names=self.COLUMNS, delimiter=',', header=0)
		self.x1 = tf.Variable(np.array(self.columns['size']).astype(np.float32))
		self.x2 = tf.Variable(np.array(self.columns['room']).astype(np.float32))
		self.y = tf.Variable(np.array(self.columns['price']).astype(np.float32))

		#数据归一化处理
		self.original_X1 = np.asarray([i[1] for i in self.data.loc[:,['size']].to_records()],dtype="float")
		self.original_X2 = np.asarray([i[1] for i in self.data.loc[:,['room']].to_records()],dtype="float")
		self.original_Y = np.asarray([i[1] for i in self.data.loc[:,['price']].to_records()],dtype="float")
		self.train_X1 = [(float(i) - np.mean(self.original_X1)) / (max(self.original_X1) - min(self.original_X1)) for i in self.original_X1]
		self.train_X2 = [(float(i) - np.mean(self.original_X2)) / (max(self.original_X2) - min(self.original_X2)) for i in self.original_X2]
		self.train_X = np.hstack((self.train_X1, self.train_X2))
		self.train_Y = [(float(i) - np.mean(self.original_Y)) / (max(self.original_Y) - min(self.original_Y)) for i in self.original_Y]

		self.n_samples = self.train_X.shape[0]
		self.X1 = tf.placeholder("float")
		self.X2 = tf.placeholder("float")
		self.Y = tf.placeholder("float")

		#初始化权重w1,w2与偏移量b
		self.W1 = tf.Variable(200.0, name="weight1")
		self.W2 = tf.Variable(200.0, name="weight2")
		self.b = tf.Variable(10.0, name="bias")

		#计算预测值
		self.sum_list = [tf.multiply(self.X1,self.W1),tf.multiply(self.X2,self.W2)] 
		self.pred_X = tf.add_n(self.sum_list)
		self.pred = tf.add(self.pred_X,self.b)

		#损失函数
		self.cost = tf.reduce_sum(tf.pow(self.pred-self.Y, 2))/(2*self.n_samples)
		
		#初始化TensorFlow优化器
		if(self.sgd_type == 'SGD'):
			self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
		
		elif(self.sgd_type == 'Momentum'):
			self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9).minimize(self.cost)

		elif(self.sgd_type == 'NAG'):
			self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,0.9,use_nesterov=True).minimize(self.cost)
		
		elif(self.sgd_type == 'Adagrad'):
			self.optimizer = tf.train.AdagradOptimizer(self.learning_rate,initial_accumulator_value=0.1).minimize(self.cost)
	
		elif(self.sgd_type == 'Adadelta'):
			self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate,rho=0.95,epsilon=1e-06).minimize(self.cost)

		elif(self.sgd_type == 'RMSPop'):
			self.optimizer = tf.train.RMSPopOptimizer(self.learning_rate,decay=0.9, momentum=0.0, epsilon=1e-10).minimize(self.cost)

		elif(self.sgd_type == 'Adam'):
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

	def run(self):
		starttime = datetime.datetime.now() #程序开始迭代时间
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(self.training_epochs):
				for (self.x1, self.x2, self.y) in zip(self.train_X1, self.train_X2, self.train_Y):
					sess.run(self.optimizer, feed_dict={self.X1: self.x1, self.X2:self.x2, self.Y: self.y})
				c=sess.run(self.cost,feed_dict={self.X1:self.train_X1,self.X2:self.train_X2,self.Y:self.train_Y})
				if (epoch+1) % self.display_step == 0:           
					print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"W1=", sess.run(self.W1), "W2=", sess.run(self.W2), "b=", sess.run(self.b))
			print("Done!")
			training_cost = sess.run(self.cost, feed_dict={self.X1:self.train_X1 , self.X2:self.train_X2, self.Y: self.train_Y})
			print("cost=", training_cost, "W1=", sess.run(self.W1), "W2=", sess.run(self.W2), "b=", sess.run(self.b), '\n')
			print ("W1=",sess.run(self.W1))
			print ("W2=",sess.run(self.W2))
			print ("b=",sess.run(self.b))
			endtime = datetime.datetime.now() #计算运行时间差值，得出运行耗时
			print ("Run time:"+str(endtime - starttime))

			#在数据集上绘制面积-房价预测函数
			plt.plot(self.original_X1, self.original_Y, 'ro', label='Original Points', color='blue')
			plt.plot(self.original_X1, sess.run(self.W1) * self.original_X1 + sess.run(self.W2) * self.original_X2 + sess.run(self.b),label='Fitted line',color='red' )
			plt.legend()
			plt.show()

if __name__ == '__main__':
	testSGD = SGD('data/data.csv')
	testSGD.run()