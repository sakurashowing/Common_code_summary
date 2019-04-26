# 1、图片读写

## 1、图片读入

```
image = cv2.imread('test.jpg', -1)
```

第一个参数文件路径，第二个参数是通道数，-1代表原图与原图保持一致，0代表转换为灰度图，1代表使用BGR通道读入。

需要注意的就是读入的图片是以np.ndarray类型存储的，且彩色图像的通道为BGR与常见的RGB不同。

## 2、图片展示

```text
cv2.imshow('image', image)
cv2.WaitKey(0)
cv2.destroyAllWindows()
```

imshow第一个参数是窗口名，string类。第二个是np.ndarray的数组。

在Ubuntu下cv2的imshow会出bug， 使程序崩溃，因此使用pyplot展示图片，但需要注意pyplot的彩色图片是RGB的格式，所以展示图像时需要先转换通道。

对于灰度图像：

```text
plt.imshow(image, cmap='gray')
plt.show()
```

对于彩色图像：

```text
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
```

## 3、图片写出：

```text
cv2.imwrite('image.jpg', image)
```

这边需要注意的是第一个参数文件名要加格式，比如.jpg .png 之类的。把image以image.jpg名称写入

## 4、More

多的话只对文件读入加一些。

读取某个文件夹下所有jpg或其他图片可以使用glob，但无法读取子文件夹下的。

```text
import glob
PATH = ''
for file in glob.glob(PATH + '*.jpg'):
    img = cv2.imread(file, -1)
```

也可以把jpg改为png等等。

那么如何读取一个文件夹内包括其子文件夹下的所有图片呢？这个时候要用os和fnmatch了。

```text
import os
import fnmatch


def find_files(directory, pattern):
    """
    Method to find target files in one directory, including subdirectory
    :param directory: path
    :param pattern: filter pattern
    :return: target file path
    """
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

for filename in find_files('Home/Leon', '*.jpg'):
    img = cv2.imread(filename, -1)
```

这个看一下应该就知道怎么回事了把？find_files引用了一个网上的代码我忘了哪个了，，，

然后对读入的图片色彩空间转换，常用cvtColor

```text
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

第二个参数很多种，常用的色彩空间也就是把BGR转成RGB或者Gray，偶尔会用HSV，因为HSV不会有很多颜色的参数，对于图片的纹理啊这些特征的提取和处理很有用。HSV三个通道分别是：色调（H），饱和度（S），明度（V）。具体释义可以看维基百科的

[https://zh.wikipedia.org/wiki/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/HSL%25E5%2592%258CHSV%25E8%2589%25B2%25E5%25BD%25A9%25E7%25A9%25BA%25E9%2597%25B4) 

实话讲对HSV我理解的不多，反正就是H跟颜色有关，是一种对于RGB的非线性变换，饱和度的话就是我们有时候会说有些颜色好艳啊之类的，我会用来提取纹理，V明度我一直就认为是类似灰度值了。印象中有专门的矩阵来乘可以完成两个色彩空间的变换。

# 2、图像滤波

## 1、图像降噪：

### 1、几种降噪方法

降噪第一步是估计噪声，只有知道是什么样子的噪声类型才可以找到好的降噪方法。估计噪声的话一般先观察直方图，也就是Histogram。一般有两种方法，一种是opencv自带的，还有一个matplotlib的，我个人倾向第二种，因为简单。

```text
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
```

对于如何估计噪声不说了，主要是写应用的。

对于降噪常用的方法有均值滤波，高斯滤波，中值滤波，双边滤波等，当然你也可以定义自己的滤波函数。

```text
blur = cv2.blur(img,(5,5))
gau_blur = cv2.GaussianBlur(img,(5,5),0)
median = cv2.medianBlur(img,5)
bi_blur = cv2.bilateralFilter(img,9,75,75)
img_nonlocal= cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
```

从上到下分别是均值滤波，高斯滤波，中值滤波，双边滤波，非局部均值滤波。下面说一下应用场景：

1. 均值滤波我反正不怎么用，了解滤波概念学习下就好了。
2. 高斯滤波一般都会有一些效果，当我实在找不到最好的滤波方法时我一般使用高斯滤波，不保证效果但不会出问题。主要滤高斯噪声的，就是直方图看着正态分布那样。但现实图片中并没有高斯噪声，高斯只是比较适合去拟合现实中的一些噪声。（不确定欢迎指正）
3. 均值滤波我用的比较多，对于椒盐噪声有很好的效果，就是直方图上会有一些孤立的高点那样的。当然滤出椒盐噪声的话维纳滤波也会有很好效果，但较为麻烦所以不怎么用。
4. 双边滤波相当于把大图片分小再一块一块滤波，很多时候效果也不错，网上看说美颜会经常用。他比较好的一点是可以保留原始图像的纹理边缘等细节。
5. 图像中的像素点之间不是孤立存在的，某一点的像素与别处的像素点一定存在某种关联，可以概括为灰度相关性和几何结构相似性。这是均值滤波考虑的地方，但是我们还发现，相似像素并不局限于某个局部区域，如图像中的长边缘，结构纹理等，都是相似的。自然图像中含有丰富的重复冗余信息，所以可以采用能够描述图像结构特征的图像块在整个图像上寻求相似块。

对于函数具体参数可以参看

[Smoothing Images](https://link.zhihu.com/?target=https%3A//opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html%3Fhighlight%3Dfilter) 

除此之外还有局部均值滤波等，non-local-means，我个人感觉和bilateralFilter比较相似，用于一张大图上有一些相似部分的滤波，速度很慢而且占用内存也挺大，不建议用。

[Image Denoising - OpenCV-Python Tutorials 1 documentation](https://link.zhihu.com/?target=https%3A//opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html%3Fhighlight%3Dfilter)

### 2、定量评估

```
#coding:utf8
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.measure import compare_psnr
#import pybm3d
filename = "noise.jpg"
gtname = "noise_clean.jpg"
img = cv2.imread(filename)
imggt = cv2.imread(gtname)

##----高斯滤波----##
plt.figure(figsize=(16,10))
img_gaussian= cv2.GaussianBlur(img,(5,5),0)
psnr=compare_psnr(imggt,img_gaussian) #计算图像的峰值信噪比
print("gauss blur psnr:%f"%psnr)
```



## 2、边缘检测：

实际应用中我用的比较多的是找边界，findContours。

```text
contours, _ = cv2.findContours(img_close, 1, 2)
```

其中contours保存了图像中所有的contour信息，是一个列表。

这里注意下opencv版本更新，老版finContours是返回三个参数，现在是两个参数。还有Contour都是闭合的。但很多时候Contour会有大有小，我们可以设置阈值来滤出一些我们不想要的边界，最常用的是根据包围的面积大小来看，偶尔也会根据周长。在滤出contour后经常会话bounding box， 因为Contour通常是不规则的。

```text
for contour in contours:
    if area_thresh_max > cv2.contourArea(contour) > area_thresh_min:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
```

boundingRect返回的分别是Contour左上角坐标x，y和矩形框的宽高w，h。

另外对于Contour还可以根据其area或len排个序，取前几位这样。

```text
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
```

此外边缘检测还经常用一些算子，Canny， Sobel， Laplacian等。其中Canny对边缘检测效果最好，但是很多时候也会受到噪音的干扰。Laplacian经常用来做边缘增强。

```
#coding:utf8
import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
filename='shizi.jpg'
image = cv2.imread(filename)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
image = cv2.resize(image,(600,400))
plt.figure(figsize=(16,10))

#sobel
sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)#x方向的梯度
sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)#y方向的梯度
sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值
sobelCombined = cv2.bitwise_or(sobelX,sobelY)
cv2.imwrite("sobel.jpg",sobelCombined)

#拉普拉斯边缘检测
lap = cv2.Laplacian(image,cv2.CV_64F,ksize = 3)#拉普拉斯边缘检测
lap = np.uint8(np.absolute(lap))##对lap去绝对值
cv2.imwrite("laplacian.jpg",lap)

#canny
canny = cv2.Canny(image, 50, 150)
cv2.imwrite("canny.jpg",canny)

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(image, -1, kernelx)
img_prewitty = cv2.filter2D(image, -1, kernely)
prewitt = cv2.bitwise_or(img_prewittx,img_prewitty)#
cv2.imwrite("prewitt.jpg",canny)

plt.subplot(141)，plt.title("sobel")，plt.imshow(sobelCombined,cmap="gray")
plt.subplot(142)，plt.title("lap")，plt.imshow(lap, cmap="gray")
plt.subplot(143)，plt.title("canny")，plt.imshow(canny, cmap="gray")
plt.subplot(144)，plt.title("prewitt")，plt.imshow(prewitt, cmap="gray")
plt.show()
```

![QQ截图20190419155046](D:\JupyterNotebook\常用代码\QQ截图20190419155046.jpg)

我建议大家可以根据实际情况自己定义kernel算子，然后使用filter2D来做。

比如求上下的边缘就可以定义：

```text
KERNEL_HORIZON = np.asarray([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=np.int)
```

然后使用：

```text
img = cv2.filter2D(img, -1, KERNEL_HORIZON)
```

左右边缘的话可以：

```text
KERNEL_VERTICAL = np.asarray([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=np.int)
```

反正算子随便定，你3X3，5X5，数值也随意，反正多试试哪个好用哪个。

不过需要注意一点是矫正，你比如我的KERNEL_HORIZON就有可能让边缘向上或向下偏一个像素值，虽然很小，但如果是做cutting的话就不是那么精确了。

还有线检测圆检测等，使用Hough方法，该方法原理比较简单，很容易懂，建议大家看看原理。但在Hough前要先将图像二值化，即非黑即白。好像是后面我要说的，可以跳着看。

```text
edges = cv2.Canny(gray_img,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
```

在实际应用中碰到的问题往往与纯粹的直线会有些偏差，所以我一般会自己仿照着Hough的原理写一下针对性的函数。所以还是看看原理吧兄弟们。

# 3、图像增强

图像增强，包含对比度增强、超分辨率、图像修复

对比度增强包含点操作、直方图操作、Retinex理论

点操作也被称为直接对⽐度增强，将每个像素独⽴操作，包括对数变化，指数变化，负图像，阈值化等。我们熟知的 gamma 变换如下，可以进⾏不同形状的映射。

![QQ截图20190419153423](D:\JupyterNotebook\常用代码\QQ截图20190419153423.jpg)



```
![QQ截图20190419154104](D:\JupyterNotebook\常用代码\QQ截图20190419154104.jpg)#Gamma变换就是用来图像增强，其提升了暗部细节，简单来说就是通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，
#即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正。
import cv2
import matplotlib.pyplot as plt
import numpy as np
filename='./qianbaoguang.png'
img = cv2.imread(filename)
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grey = cv2.resize(grey,(600,400))
def Residual(residual,size,title):
    plt.subplot(size)
    plt.title(title)
    plt.imshow(residual)
plt.figure(figsize=(12,6))
Residual(img,231,'img')
#第一种写法
paras=[1.0,2.0,3.0,4.0,5.0]
#enumerate:序列对应，0—1.0，1-2.0
for i,para in enumerate(paras):
    Residual(np.power(img/float(np.max(img)), 1/para),232+i,'1/%.1f'%para)   
plt.tight_layout()#自动调整子图参数，使之填充整个图像区域
#第二种写法
paras=[1.0,2.0,3.0,4.0,5.0]
plt.figure(figsize=(12,6))
plt.subplot(231)
plt.title("org pic")
plt.imshow(img)
#变换公式就是对原图像上每一个像素值做乘积运算
for i, para in enumerate(paras):
    plt.subplot(232+i)
    plt.title('1/%.1f'%para)
    #np.power求1/para的次方
    img1 = np.power(img/float(np.max(img)), 1/para)
    plt.imshow(img1)
plt.tight_layout()
plt.show()
```

![QQ截图20190419154104](D:\JupyterNotebook\常用代码\QQ截图20190419154104.jpg)

图片拼成一幅图

```
%matplotlib inline
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./guobaoguang.jpeg',0)
# histogram equalization
equ = cv2.equalizeHist(img)
# 两个图片的像素分布连接在一起，拍成一维数组
res = np.hstack((i图片拼成一幅图mg,equ))
```

![QQ截图20190419154729](D:\JupyterNotebook\常用代码\QQ截图20190419154729.jpg)

还是先看一眼直方图然后决定怎么增强吧。最简单的无外乎线性变换，比如你原图的直方图灰度分布0-128就到顶了，那就可以img = img*2，对比度就增强了两倍。这也是全局化的增强，叫直方图均衡化（Histogram Equalization）。

OpenCV里有直接的函数

```text
img_equ = cv2.equalizeHist(img)
```

还有局部自适应直方图均衡化，叫CLAHE（Contrast Limited Adaptive Histogram Equalization），这也是我用的比较多的。

```text
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)
```

第一个参数是对比度的limited，第二个参数是局部的窗口大小。官方文档也写得很清晰了：

[Histograms - 2: Histogram Equalization](https://link.zhihu.com/?target=https%3A//opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html%3Fhighlight%3Dhistogram) 

增强这一部分其实很多类，上面说的都是对比度的增强，当然还有边缘增强，就是锐化，这种一般先使用边缘检测提取出边缘的特征图，再拿边缘特征图与原图相加就可以是边缘部分得到增强。常用的是高斯边缘检测。

# 4、阈值分割

## 1、基本阈值分割

```python
#原图像矩阵、分类的阈值、高于（低于）阈值赋予的新值、方法选择参数,返回值为阈值、处理后的图像矩阵
ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)#黑白二值
ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)#黑白二值反转
ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)#多像素值
ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)#高于阈值时像素为255
ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)#低于阈值时像素为255
```

## 2、自适应阈值

```python
#自适应阈值
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th4=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)
th5=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
th6=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
```

阈值分割其实也就是二值化，首先我觉得大家都知道现在几乎所有图片都是256个灰度级吧，二值化就是变成非黑即白，就是只有两个灰度级。

最简单的阈值分割就是，比如127这个灰度吧，你让大于127的变成255，小于等于127的变成0就好了，这个可以直接用numpy的数据结构操作的，不需要用什么OpenCV的函数，反正建议大家尽量不要过于依赖OpenCV已有的函数，要活学活用是吧。

```text
img[img > 127] = 255
img[img < 128] = 0
```

可以看出numpy是很方便的，需要说明下这里最好不要用两个for循环来遍历图片，比大小，然后赋值0或255，这样速度太慢了。numpy里是矩阵运算的，速度会快很多。

上面这个其实就是OpenCV里的THRESH_BINARY，里面还有inverse就是黑白换一下，这玩意直接拿255减原图不就妥了嘛，反正简单的二值化建议直接用numpy做。

OpenCV里还有Adaptive Threshold，有两个参数分别是mean和gaussian，反正我理解的就是，先做个均值滤波或者高斯滤波完了再简单的Threshold，所以我也不怎么用。不过还是放一下文档链接吧，随缘看。

[Image Thresholding](https://link.zhihu.com/?target=https%3A//opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html%3Fhighlight%3Dthreshold) 

## 3、HSV转换空间

```python
##------转换空间再求阈值------##
rgbimg=cv2.imread(filename)
rgbimg = cv2.resize(rgbimg,(600,400))
hsvimg = cv2.cvtColor(rgbimg,cv2.COLOR_BGR2HSV)
plt.figure(figsize=(12,6))
plt.subplot(131)
plt.imshow(hsvimg[:,:,0],cmap ='gray')
plt.title("h")
plt.subplot(132)
plt.imshow(hsvimg[:,:,1],cmap ='gray')
plt.title("s")
plt.subplot(133)
plt.imshow(hsvimg[:,:,2],cmap ='gray')
plt.title("v")
plt.show()
grey = hsvimg[:,:,2]
greyblur = cv2.GaussianBlur(grey,(5,5),0)
th,result = cv2.threshold(greyblur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```

![QQ截图20190419145223](D:\JupyterNotebook\常用代码\QQ截图20190419145223.jpg)

## 4、OTSU二值化

OK，上面的都不怎么用那我阈值分割平常用什么啊？我觉得利器就是OTSU，这种阈值分割是自适应的阈值，会根据Histogram计算出最小方差差的灰度值，然后用那个灰度值来做简单的阈值分割。这种对于直方图有个波浪起伏那种特别好用，就是下面第三个。因为很多时候我们很难去确定到底用哪个阈值，比如上面说127是随意说的，对不同图片就阈值不一样。

![img](https://pic2.zhimg.com/80/v2-cf4b540278fc89f0cd52ad8b16778435_hd.jpg)

也不一定是完全要求这样非常明显的分布啦，比如第二个的直方图，只要后半部分有那么一点点上升的趋势，就可以很好的分出来了。用起来也很好用：

```text
ret1,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)#黑白二值
ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```

ret我记得是算出来的那个阈值，th就是ndarray的图片了，一般我只关心图片，所以就直接

```text
_, th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```

就很舒服。同样的，对于这是针对全局Histogram只有一个凹槽的OTSU，那么如果Histogram波浪起伏呢？这个时候就可以分段来提取了，比如我在0-127做一遍OTSU，阈值下0阈值上我给127， 然后再对剩下的继续OTSU，就这样。前提是需要提前观察好图像的特征，所有方法都不是死搬硬套的，要针对具体问题来改变。

## 4、解决matplotlib中文无法显示问题

```
#解决matploylib中中文无法显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.hist(img.ravel(),256)
plt.title('灰度直方图')
plt.show()
```

# 5、形态学操作

## 1、膨胀和腐蚀

形态学操作也是我用的比较多的。讲出这个概念，大概第一反应就要想到膨胀和腐蚀了，在OpenCV里函数直接就是dilate和erode。除了膨胀腐蚀还有开运算和闭运算，说白了就是开运算：先腐蚀后膨胀，闭运算：先膨胀后腐蚀。关于膨胀腐蚀的概念和源码可以看看：

[腐蚀与膨胀(Eroding and Dilating)](https://link.zhihu.com/?target=http%3A//www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html)

我当时学的时候到这里就很疑惑，为什么有了膨胀腐蚀还要开运算闭运算呢？其实开闭运算最重要的一点就是，可以保持物体原有大小。然后一个是消除物体内部孔洞的另一个是增强物体之间连接点的，当然要根据背景是黑是白来确定用那个，用0还是1的kernel等等。

```text
kernel = np.ones((10, 10), dtype=np.uint8)
img = cv2.erode(img, kernel=kernel, iterations=1)
img = cv2.dilate(img, kernel=kernel, iterations=1)
```

很多人知道filter2D的kernel必须要奇数，但是这里形态学并不是。首先明确形态学操作也是对二值图进行的，简单来讲就是你图上这一块和我的kernel都一样我就置为255，不一样我就置为0，当然erode和dilate是相反的，其实跟背景也有很多关系，所以有时候我会：

```text
img = cv2.erode(255-img, kernel=kernel, iterations=1)
```

就不用想太多，show一下看看不行了再调回去。然后iteration就是指迭代几次，比如iterations=2下面两行是等价的：

```text
img = cv2.erode(img, kernel=kernel, iterations=2)
img = cv2.erode(cv2.erode(img, kernel=kernel, iterations=1), kernel=kernel, iterations=1)
```

然后开闭运算就是：

```text
img = cv2.dilate(cv2.erode(img, kernel=kernel, iterations=1), kernel=kernel, iterations=1)
img = cv2.erode(cv2.dilate(img, kernel=kernel, iterations=1), kernel=kernel, iterations=1)
```


是不是很简单？其实OpenCV里还有个专门的morphology来进行开运算闭运算，据说有一些改变，效果会好一些（一个大佬建议我尽量用morphology的），反正我是一直觉得差不多。

```text
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

除此之外open和close还有一点用就是也可以作为边界的提取。就是拿Open的图片减去close的图片就好了，morphology里也有专门的参数，反正我还是觉得下面两个等价的。

```text
edges = opening - closing
edges = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

有时候我也会设置一些奇葩的kernel比如不全1啊等等，还有可以用不同大小的kernel来分别erode和dilate，我就用过10X1的kernel先膨胀再用1X10的腐蚀回来，当时是为了消除一些水平线我记得。

## 2、区域生长（Region Growing）-->Watershed

这也是用于图像分割的非监督学习算法，通用的解释都是把这个算法等效成高低起伏的山山上还有些小坑那种。

你想啊，好多个连着的山，一个挨着一个，这个时候采用传统阈值分割，就很难把山和山分开（采用聚类的K-means应该还行），但是要注意有山峰还有山谷啊，还有那些坑对吧。假想有一场一直下着的雨，那么水往低处流，水填充的地方就可以很好的把山和山给分开，甚至于山上的一些小坑也能分的出来。

这里山的陡峭程度就是图像的梯度，就是图像变化特别剧烈的部分。唉口语怎么感觉说不清呢？直接引一下百度百科吧：

在该算法中，分水岭计算分两个步骤，一个是排序过程，一个是淹没过程。首先对每个像素的灰度级进行从低到高排序，然后在从低到高实现淹没过程中，对每一个局部极小值在h阶高度的影响域采用先进先出(FIFO)结构进行判断及标注。

就是，想象一个二维的图，你把它想成三维的，xy轴然后每个坐标点的灰度值当z轴，有画面了吧，一个高低起伏的三维立体图。然后你就想象拿一壶水网这图里倒，就行了。水位线以下一点距离再往下的给个label -1（背景）， 水位线以上一点距离再往上给个label 1（前景），水位线附近的给个label 0（边界）。然后吧label等于某一值的赋值就好了。类似这样：

![img](https://pic3.zhimg.com/80/v2-da252e80a960a4e6351608955b1a8232_hd.jpg)

```text
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
```

注意一点不要把读入的图片直接拿来用，最好先预处理下，腐蚀膨胀下之类的，效果会好很多。

## 3、Super Pixel

这个概念其实是很类似于机器学习里的pooling操作的，主要目的是为了在保留图像大致信息的情况下缩小图像的尺寸，其实也不绝对。反正看看下面这两个图片大概就明白了吧。

![QQ截图20190418235419](D:\JupyterNotebook\常用代码\QQ截图20190418235419.jpg)

上面那个图可以抽象为下面这个图，而且绝大部分的信息（包括边界，形状，颜色等等）都被保留了，同时还可以相当于缩小了大小（把那些看着块状的区域直接替代下就行了）。有感觉 了吧，superpixel其实在传统cv里没什么用（我感觉），在CNN等机器学习网络里常用pooling也就是池化层来实现，比如maxpooling，averagepooling，网上有很多相关的资料。下图放一个maxpooling的吧。

![img](https://pic2.zhimg.com/80/v2-3d8a6782133970cc1d53c40963303ae1_hd.jpg)

这个是步长为二的最大值池化，就是2*2里面选最大的像素点的灰度值来代替这四个像素点的坐标值。以此类推。pooling步长的选择一般都是2，太大的话会丢失很多信息，再小，，1有个p用啊，，，所以就2了一般。

# 6、实战

## 1、图片分割

### 1、⽆监督的灰度图像的⾃动前背景分离

```
import matplotlib.pyplot as plt
import cv2
import numpy as np

filename='./flower/600.jpg'
img=cv2.imread(filename)
print(img.shape[:2])
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
plt.show()
ret,th1=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
plt.imshow(th1,cmap='gray')
plt.title('otsu')
plt.show()
```



### 2、有监督的彩⾊图的前背景分离

```
filename='./flower/600.jpg'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask=np.zeros(img.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(50,0,300,250)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
filename='./flower/600.jpg'
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask=np.zeros(img.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(50,0,300,250)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
```

![QQ截图20190419155926](D:\JupyterNotebook\常用代码\QQ截图20190419155926.jpg)

### 3、代码解释

```
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('varese.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

# zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (1, 1, img.shape[1], img.shape[0])
# 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
# img - 输入图像
# mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
# rect - 包含前景的矩形，格式为 (x,y,w,h)
# bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
# iterCount - 算法的迭代次数
# mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

# np.where 函数是三元表达式 x if condition else y的矢量化版本
# result = np.where(cond,xarr,yarr)
# 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
# | 是逻辑运算符or的另一种表现形式
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# mask2[:, :, np.newaxis] 增加维度
img = img * mask2[:, :, np.newaxis]

# 显示图片
plt.subplot(121), plt.imshow(img)
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('varese.jpg'), cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])
plt.show()
```

### 4、完整图像分割

Opencv项目实战-图像分割思路：使用otsu方法在HSV颜色空间完成预分割，将结果作为前背景的可能的标注，再使用grabcut方法完成分割

```
#coding:utf8
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mplimage
import numpy as np
import cv2

import math

class ImageSeg:
    def __init__(self,color_space='GRAY'):
        print ("init")
        self.color_space = color_space
        
    def otsu_seg(self,rgbimg):
        if self.color_space == 'GRAY':
            greyimg = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
            greyblurimg = cv2.GaussianBlur(greyimg,(5,5),0)
            self.th,self.otsumask = cv2.threshold(greyblurimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif self.color_space == 'HSV':
            hsvimg = cv2.cvtColor(rgbimg,cv2.COLOR_BGR2HSV)
            greyimg = hsvimg[:,:,2]
            greyblurimg = cv2.GaussianBlur(greyimg,(5,5),0)
            self.th,self.otsumask = cv2.threshold(greyblurimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
        elif self.color_space == 'Lab':
            labimg = cv2.cvtColor(rgbimg,cv2.COLOR_BGR2Lab)
            greyimg = labimg[:,:,1]
            greyblurimg = cv2.GaussianBlur(greyimg,(5,5),0)
            self.th,self.otsumask = cv2.threshold(greyblurimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

    def grabcut(self,image):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (100,100,500,400)
        mask = np.zeros(image.shape[:2],np.uint8)
        mask[self.otsumask == 0] = cv2.GC_PR_BGD
        mask[self.otsumask == 255] = cv2.GC_PR_FGD
        mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        self.grabcutmask = np.where((mask==2)|(mask==0),0,1).astype('uint8')*255
    #image = image*mask[:,:,np.newaxis]

    def process(self,image):
        ##1,获得二值分割结果
        height,width,channel = image.shape   
        self.otsu_seg(image)
        self.grabcut(image)
        return self.otsumask,self.grabcutmask
```

```
myimageseg = ImageSeg('HSV')
images = os.listdir('./flower/')
for imagename in images:
    if imagename.endswith('.jpg'):
        imagepath = os.path.join('./flower/',imagename)
        image = cv2.imread(imagepath)
        print ("processing:",imagepath)
        otsuresult,grabcutresult = myimageseg.process(image)
        cv2.imwrite(imagepath.replace('.jpg','_hsv_otsu.png'),otsuresult)
        cv2.imwrite(imagepath.replace('.jpg','_hsv_grab.png'),grabcutresult)
        cv2.namedWindow("image",0)
        cv2.imshow("image",image)
        cv2.namedWindow("otsu",0)
        cv2.imshow("otsu",otsuresult)
        cv2.namedWindow("grab",0)
        cv2.imshow("grab",grabcutresult)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
```



## 2、车道检测

### 1、hough变换

```
#coding: utf8
import cv2
import numpy as np
import matplotlib.pyplot as plt

##首先检测边缘
img = cv2.imread('lane.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaus = cv2.GaussianBlur(gray, (5, 5), 0)
lowth = 50
highth = lowth * 3
edges = cv2.Canny(gaus, lowth, highth, apertureSize=3)
#cv2.namedWindow("edge",0)
#cv2.imshow("edge",edges)
#cv2.waitKey(0)
plt.figure(figsize=(16,10))
plt.subplot(121)
plt.imshow(edges,cmap='gray')
plt.title('edges')

##然后检测直线
height,width=gray.shape
minLineLength = 100
maxLineGap = 30
lines = cv2.HoughLines(edges,1,np.pi/180,200) #返回三个值，像素长度和弧度
print(lines)
print ("gray image size=",gray.shape)
print ("lines[0]",lines[0])

##1 返回半径大小和角度##
for rho,theta in lines[0]:
    print(theta)
    a = np.cos(theta) ##x方向偏移
    print(a)
    b = np.sin(theta) ##y方向偏移
    print(b)
    x0 = a*rho
    y0 = b*rho
    
    ##右无穷点
    x1 = int(x0 + 1000*(-b)) 
    y1 = int(y0 + 1000*(a))
    ##左无穷点
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    print ("x0=",x0,"x1=",x1,"x2=",x2)
    print ("y0=",y0,"y1=",y1,"y2=",y2)

    cv2.circle(img,(int(x0),int(y0)),4,(255,255,0),2,8,0) ##画出切点
    cv2.line(img,(0,0),(x0,y0),(255,0,0),2) ##画出半径
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2) ##画出直线
    

#cv2.imshow("edges", img)
#cv2.imwrite("houghlie.jpg",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.subplot(122)
plt.imshow(img)
plt.title('img')
```

![QQ截图20190419160303](D:\JupyterNotebook\常用代码\QQ截图20190419160303.jpg)

```
import numpy as np
import cv2
import matplotlib.pyplot as plt

lane = cv2.imread('./lane.jpg')
gray = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
# 高斯模糊，Canny边缘检测需要的
lane1 = cv2.GaussianBlur(gray, (5, 5), 0)
# 进行边缘检测，减少图像空间中需要检测的点数量
lane2 = cv2.Canny(lane1, 50, 150)
#cv2.imshow("lane", lane2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.figure(figsize=(16,10))
plt.subplot(121)
plt.imshow(lane2,cmap='gray')
plt.title('edges')

rho = 1  # 距离分辨率
theta = np.pi / 180  # 角度分辨率
threshold = 30  # 霍夫空间中多少个曲线相交才算作正式交点
min_line_len = 50  # 最少多少个像素点才构成一条直线
max_line_gap = 5  # 线段之间的最大间隔像素
lines = cv2.HoughLinesP(lane2, rho, theta, threshold,minLineLength=min_line_len, maxLineGap=max_line_gap)
lines3 = lines[:,0,:]#提取为二维
for x1,y1,x2,y2 in lines3[:]: 
    cv2.line(lane,(x1,y1),(x2,y2),255,1)

plt.subplot(122)
plt.imshow(lane)
plt.title('lane')

```

![QQ截图20190419160414](D:\JupyterNotebook\常用代码\QQ截图20190419160414.jpg)

### 2、完整代码

```
#coding:utf8
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mplimage
import numpy as np
import cv2
import math



# Hough transform parameters
#rho:半径步长
#theta:最小弧度
#threshold:累加器中的值高于它时才认为是一条直线
#min_line_length:线的最短长度，比这个短的都被忽略
#max_line_gap:两条直线之间的最大间隔，小于此值，认为是一条直线

class LaneDetect:
    def __init__(self,canny_params,hough_params):
        self.canny_params = canny_params
        self.hough_params = hough_params

    def filter_lines(self,lines,angle):
        ##去除斜率过大的水平线
        #y2-y1>tan(angle*3.1415/180)*(x2-x1)
        thelines = [line for line in lines if (abs(line[0][3]-line[0][1]) > math.tan(angle*3.1415/180)*abs(line[0][2]-line[0][0]))]
        return thelines 
    
    def group_lines(self,lines,mergeangle):
        ##计算斜率
        lines = np.asarray(lines)
        slopes = np.asarray([(line[0][3]-line[0][1])/(line[0][2]-line[0][0]+0.000001) for line in lines])
        print ("type slope=",type(slopes),"type lines",type(lines))
        ##rank
        print ('sort=',np.argsort(slopes))
        print ('type lines and index',type(lines),type(np.argsort(slopes)))
        print ('shape',(np.argsort(slopes)).shape)

        sortedlines = list(lines[np.argsort(slopes)])
        print(sortedlines)
        mergeth = math.tan(mergeangle*3.1415/180) ##0.087,5度以内的线段合并
        iniline = sortedlines[0]
        i = 1
        inislope = (iniline[0][3]-iniline[0][1])/(iniline[0][2]-iniline[0][0]+0.000001)
        #inilength = (iniline[0][3]-iniline[0][1])/(iniline[0][2]-iniline[0][0]+0.000001)
        inilength=(iniline[0][3]-iniline[0][1])*(iniline[0][3]-iniline[0][1])+(iniline[0][2]-iniline[0][0]+0.000001)*(iniline[0][3]-iniline[0][1])
        while(len(sortedlines)):
            if i >= len(sortedlines):
                break
            curline = sortedlines[i]
            curslope = (curline[0][3]-curline[0][1])/(curline[0][2]-curline[0][0]+0.000001)
            curlength = (curline[0][3]-curline[0][1])*(curline[0][3]-curline[0][1])+(curline[0][2]-curline[0][0]+0.000001)*(curline[0][3]-curline[0][1])
            if abs(curslope-inislope) < mergeth:
                if curlength > inilength:
                    inislope = curslope
                    inilength = curlength
                    sortedlines.pop(i-1)
                else:
                    sortedlines.pop(i)
                i = i - 1
            else:
                inislope = curslope
                inilength = curlength
            i = i + 1
        return sortedlines 

    def process(self,image):
        ##1,获得ROI区域
        height,width,channel = image.shape   
        ##限定用于检测的区域，取图像的下半部分 
        xmin = 0
        xmax = width
        ymin = int(height*2/3)
        ymax = height

        ##2,边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (self.canny_params["blur_ksize"], self.canny_params["blur_ksize"]), 0, 0)
        edges = cv2.Canny(blur_gray[ymin:ymax,xmin:xmax],self.canny_params["low_threshold"],self.canny_params["high_threshold"]) 
        mask = np.zeros((height,width),np.uint8)
        print (edges.shape)
        mask[ymin:ymax,xmin:xmax]=edges

        cv2.namedWindow("test",0)
        cv2.imshow("test",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        ##3,直线检测,获得左右两侧直线
        print ("mask.type",mask.shape)
        # HoughLinesP输出的直接就是直线点的坐标位置
        lines = cv2.HoughLinesP(mask, self.hough_params["rho"], self.hough_params["theta"], self.hough_params["threshold"],np.array([]),
                                minLineLength=self.hough_params["min_line_length"], maxLineGap=self.hough_params["max_line_gap"])
        print ("detect lines",len(lines))
        left_lines, right_lines = [], []
        for line in lines:#获得左右两侧车道线
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                if k < 0:
                    left_lines.append(line)
                else:
                    right_lines.append(line)
        
        ##4,直线过滤,根据斜率进行过滤
        print ("过滤前 左侧候选车道线段数量:",len(left_lines))
        #print(left_lines)
        print ("过滤前 右侧候选车道线段数量:",len(right_lines))
        left_lines = self.filter_lines(left_lines, 10)
        left_lines = self.group_lines(left_lines, 5)
        right_lines = self.filter_lines(right_lines, 10)
        right_lines = self.group_lines(right_lines, 5)
        print ("过滤后 左侧候选车道线段数量:",len(left_lines))
        print ("过滤后 右侧候选车道线段数量:",len(right_lines))

        for left_line in left_lines:
            x1,y1,x2,y2 = left_line[0]
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        for right_line in right_lines:
            x1,y1,x2,y2 = right_line[0]
            cv2.line(image,(x1,y1),(x2,y2),(0,255,255),2)
            
        cv2.imwrite("lane_detecting.jpg",image)
        cv2.imshow("lane", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

```
canny_params = {"low_threshold":50,"high_threshold":150,"blur_ksize":5}
hough_params = {"rho":1,"theta":np.pi/180,"min_line_length":20,"max_line_gap":15,"threshold":100}
image=cv2.imread('./lane.jpg')
mylanedetect = LaneDetect(canny_params,hough_params)
mylanedetect.process(image)
```

