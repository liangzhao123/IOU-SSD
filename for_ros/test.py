import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import LaserScan

def callback(data):
    # 设置画布为600*600像素
    frame = np.zeros((600, 600,3), np.uint8)
    angle = data.angle_min
    for r in data.ranges:
        #change infinite values to 0
        #如果r的值是正负无穷大，归零
        if math.isinf(r) == True:
            r = 0
        #convert angle and radius to cartesian coordinates
        #这里就是将极坐标的信息转为直角坐标信息，只是像素的转化，不对应具体值
        #如果x中的90是正的，则顺时针显示，如果是负的，则逆时针显示。
        x = math.trunc((r * 50.0)*math.cos(angle + (-90.0*3.1416/180.0)))
        y = math.trunc((r * 50.0)*math.sin(angle + (-90.0*3.1416/180.0)))

        #set the borders (all values outside the defined area should be 0)
        #设置限度，基本上不设置也没关系了
        if y > 600 or y < -600 or x<-600 or x>600:
            x=0
            y=0
       # print "xy:",x,y
		# 用CV2画线，位置在(300,300),和目标点，颜色是(255,0,0),线宽2
        cv2.line(frame,(300, 300),(x+300,y+300),(255,0,0),2)
		# 角度得增加
        angle= angle + data.angle_increment
        # 画个中心圆
        cv2.circle(frame, (300, 300), 2, (255, 255, 0))
        cv2.imshow('frame',frame)
        cv2.waitKey(1)

# 话题监听函数，其实主要是一个初始化过程
# 初始化节点，'laser_listener' 挺重要的，是节点（node）名称，一个py文件只能有一个。
# 代表开启一个进程！匿名参数也默认是这个
# 订阅的语句。先是你需要订阅的话题要对，然后是数据类型，然后是回调函数名字
# 最后的队列很重要，不管发布还是订阅，都有queue_size参数；
# 如果默认的话，发布的频率远高于你处理图片的速度，因此根本无法实时的显示
# 所以需要换为1，只接收最新的一个消息，其他的都丢了不管~
def laser_listener():
    rospy.init_node('laser_listener', anonymous=True)
    rospy.Subscriber("lidar/scan", LaserScan,callback,queue_size = 1)
    rospy.spin()

if __name__ == '__main__':
    laser_listener()