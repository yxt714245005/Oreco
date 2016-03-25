一个基于caffe和OpenCV，高效、精确的OCR(图片文字识别)工具,支持多图片并发处理。

环境要求
Python 2.7
Caffe latest
redis 3.x
OpenCV3.x

#### 1、基础配置
修改config.py，最起码你要修改CAFFE_ROOT和REDIS配置

#### 2、开关服务
```
python2.7 bin/rec-ctl.py start|stop|restart
```


#### 3、调用方式
命令行直接调用(你可以把recognizer放到任何你想要的地方)
```
python2.7 recognizer.py img1 img2 img3 ...
```

api调用
``` python
import sys
sys.path.insert(0, 'recognizer path')
from recognizer import get_text

image_path_list = [...]
ret = get_text(image_path_list)
print ret
```

#### 4、使用效果
单图片识别效果
![](http://i.imgur.com/edQw0vq.jpg)

![](http://i.imgur.com/4IYSypZ.png)

多图片识别效果
![](http://i.imgur.com/lV647FE.png)

![](http://i.imgur.com/jwHbDQE.jpg)

![](http://i.imgur.com/KI0Fz5a.jpg)

![](http://i.imgur.com/5PIr7R6.jpg)

![](http://i.imgur.com/YVME7yh.png)