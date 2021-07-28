import numpy as np
import  matplotlib.pyplot as plt #类似 MATLAB 中绘图函数的相关函数
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1 as tf #导入tensorflow，并且使用低版本
tf.compat.v1.disable_eager_execution() #确保Session中的run()函数可以正常运行

count=100
data=[]

for i in range(count):
    x1=np.random.normal(0.00,0.55)
    y1=x1*0.1+0.3+np.random.normal(0.00,0.03)
    data.append([x1,y1])
    
x_data=[v[0] for v in data]
y_data=[v[1] for v in data]

plt.scatter(x_data,y_data,c='r')
plt.show()

w=tf.Variable(tf.random_uniform([1],-1.0,1.0),name='w')
b=tf.Variable(tf.zeros([1]),name='b')

y=w*x_data+b

loss=tf.reduce_mean(tf.square(y-y_data),name='loss')

#way=tf.train.GradientDescentOptimizer(0.5)

train=tf.train.GradientDescentOptimizer(0.1).minimize(loss,name='train')

sess=tf.Session()

init=tf.global_variables_initializer()
sess.run(init)


for step in range(100):
    sess.run(train)
    
    
temp_1=sess.run(w)
temp_2=sess.run(b)

print("y=",temp_1[0],"x+",temp_2[0],",loss=",sess.run(loss))

      
plt.scatter(x_data,y_data,c='r')
plt.plot(x_data,sess.run(w)*x_data+sess.run(b))
plt.show()






