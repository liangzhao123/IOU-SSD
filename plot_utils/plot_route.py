from folium import plugins
import folium
import os
ROOT_DIR = "/home/liang/for_ubuntu502/PVRCNN-V1.1/"
m = folium.Map([30.516573, 114.35514], zoom_start=10) #中心区域的确定

# route = folium.PolyLine( #polyline方法为将坐标用线段形式连接起来
#  location, #将坐标点连接起来
#  weight=3, #线的大小为3
#  color='orange', #线的颜色为橙色
#  opacity=0.8 #线的透明度
# ).add_to(m) #将这条线添加到刚才的区域m内

m.save(os.path.join(ROOT_DIR,"output","figures","map1.html")) #将结果以HTML形式保存到桌面上