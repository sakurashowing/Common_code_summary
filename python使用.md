# 1、Numpy使用

## 1、三维数组降为二维

```python
import numpy as np
a=np.array([[[129,374,229,369]]])

print(a[:,0,:])
out-[[129 374 229 369]]
print(a[0,:,:])
out-[[129 374 229 369]]
print(a[:,:,0])
out-[[129]]
```

```python
>>>a=np.reshape(np.arange(18),(3,3,2))
>>> a
array([[[ 0,  1],
        [ 2,  3],
        [ 4,  5]],

       [[ 6,  7],
        [ 8,  9],
        [10, 11]],

       [[12, 13],
        [14, 15],
        [16, 17]]])
>>>a=reshape(a,(-1,3))
>>>a
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11],
       [12, 13, 14],
       [15, 16, 17]])

>>>a=np.reshape(np.arange(18),(2,3,3))
>>>a
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>>a=np.reshape(a,(-1,3))
>>>array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11],
       [12, 13, 14],
       [15, 16, 17]])
```

对reshape()里面的数组形状第一个为-1，第二个为第二维元素的数目。那么就会对原来的所有元素进行一个平均分配得到一个二维数组。

## 2、np.asarray与np.array

```python
# 将list转换为ndarray
a = [1, 2]
print(np.asarray(a))    # array([1, 2])

# 如果对象本身即为ndarray，且不改变dtype，则不会copy之
a = np.array([1, 2])
print(np.asarray(a) is a)    # True

# 如果对象本身即为ndarray，且改变dtype，则还是会copy之
a = np.array([1, 2], dtype=np.float32)
print(np.asarray(a, dtype=np.float32) is a)    # True
print(np.asarray(a, dtype=np.float64) is a)    # False
```

## 3、A for B in C

```
y= [x*x for x in range(10)]
print(y)
```

输出的是一个列表：

[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

这段代码类似的就是：

```
for x in range(10):
    print(x*x)
```

列表推导公式：c for b in a : 
首先for b in a 很好理解：我们从a里面取出每一个迭代对象b。然后我们对这个b进行c操作，并且以c操作后的形式输出一个列表。所以上例实现的就是输出一个从0~99的平方数的列表。

 按照这个理解，我们同样也可以遍历生成一个2D坐标数。

```
t=  [(x,y) for x in range(2) for y in range(3)]
print(t)
```

输出的就是：

[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

和上述一致，这段代码类似的就是

```
t=[]
for x in range(2):
    for y in range(3):
        t.append((x,y))
print(t)
```

从右往左看，首先y取0值，往左边走，x此时取值为0，在往左边走，就直接输出（x,y）也就是（0，0）了

 第二次，y取值为1，x此时不变（等y变完才会改变）....以此类推

## 4、np.argsort

numpy.**argsort**(a, axis=-1, kind=’quicksort’, order=None) 
`功能:` 将矩阵a按照axis排序，并返回排序后的下标 
`参数:` a:输入矩阵， axis:需要排序的维度 
`返回值:` 输出排序后的下标

```
>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])
```

```
>>> x = np.array([[1, 5, 7], [3, 2, 4]])
>>> np.argsort(x, axis=0)
array([[0, 1, 1],
       [1, 0, 0]])  #沿着行向下(每列)的元素进行排序
>>> np.argsort(x, axis=1)
array([[0, 1, 2],
       [1, 0, 2]])  #沿着列向右(每行)的元素进行排序
```

## 5、is与==区别

1.== 是比较两个对象的内容是否相等，即两个对象的“值“”是否相等，不管两者在内存中的引用地址是否一样。

```
1.地址一样，值也一样。所以==成立。
st1 ='aaaaa'
st2 = 'bbbbb'
st3 = 'bbbbb'
st4 = st3
print(st1==st2,st2==st3,st3==st4)#False True True
print(id(st2)==id(st3),st2==st3) #True True  
 
2.引用地址不一样，但是只要值一样即可==成立。
>>> val1 = 2000
>>> val2 = 2001
>>> val3 = val1 +1
>>> print(id(val3)==id(val2),val3==val2)
False True
 
3.对于类的实例比较
class Student(object):
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def run(self):
        print("can run")
 
stu1 = Student("tom",19)
stu2 = Student("tom",19)
stu3 = stu2
print(id(stu1)==id(stu2),stu1 == stu2)  #False False
#注意这里stu1和stu2的值是不等的，虽然初始化创建对象格式一样。
print(id(stu2)==id(stu3),stu2 == stu3)  # True True 
```

2.is 比较的是**两个实例对象是不是完全相同，它们是不是同一个对象，占用的内存地址是否相同。即is比较两个条件：1.内容相同。2.内存中地址相同**

```
1.is成立的前提要是内容相同，内存中地址相同
st1 ='aaaaa'
st2 = 'bbbbb'
st3 = 'bbbbb'
st4 = st3
print(st1 is st2, st2 is st3,st3 is st4)#False True True
print(id(st1),id(st2),id(st3),id(st4))
#2439382008080 2439382008192 2439382008192 2439382008192
 
2.光值相同不同，内存地址也要相同，才会成立。
>>> a = 1
>>> a = 1000
>>> b = 1000
>>> print(id(a),id(b))
2625727620144 2625727619248
>>> print(a is b)
False
>>> print(a ==b)
True
>>>
 
3.类实例的比较，也要内存地址一致。
class Student(object):
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def run(self):
        print("can run")
 
stu1 = Student("tom",19)
stu2 = Student("tom",19)
stu3 = stu2
print(id(stu1),id(stu2),id(stu3))
print(stu1 is stu2,stu2 is stu3)
=========================================
2091922655776 2091922655888 2091922655888
False True
```

3.使用is注意python对于小整数使用对象池存储问题

```
1.举个例子，在python命令行模式下：为什么同样值a,b与c,d的结果却不一样呢？
>>> a = 1000
>>> b = 1000
>>> a is b
False
>>> c = 10
>>> d = 10
>>> c is d
True
 
注意，因为python对小整数在内存中直接创建了一份，不会回收，所有创建的小整数变量直接从对象池中引用他即可。
但是注意Python仅仅对比较小的整数对象进行缓存（范围为范围[-5, 256]）缓存起来，而并非是所有整数对象。
也就说只有在这个[-5,256]范围内创建的变量值使用is比较时候才会成立。
 
>>> e ,d ,f ,g = -5 ,-5 ,-6 ,-6
>>> e is d
True
>>> f is g  #超过-5的范围不成立
False
>>>
```

**注意：注意：上面对于python小整数对象池的使用仅仅是在命令行中执行可以，而在Pycharm或者保存为文件执行，结果是不一样的，这是因为解释器做了一部分优化。下面使用pycharm,即使整数超过256，使用is也是成立的。**

4.使用is注意python关于字符串的intern机制存储 

注意python中创建两个内容一样的变量时（变量名不一样），一般都会在内存中分配两个内存地址分别给这两个变量。即两个变量的内容虽然样，但是变量的引用地址不一样。所以两个变量使用==比较成立，但是使用 is比较不成立。

但是在python中有两个意外情况：
1.使用python命令行时对于小整数[-5,256]区间内的整数,python会创建小整数对象池，这些对象一旦创建，就不会回收，所有新创建的在这个范围的整数都是直接引用他即可。所以造成在[-5,256]区间内的整数不同变量只要值相同，引用地址也相同。此范围外的整数同样遵循新建一个变量赋予一个地址。

2.python中虽然字符串对象也是不可变对象,但python有个intern机制，简单说就是维护一个字典，这个字典维护已经创建字符串(key)和它的字符串对象的地址(value),每次创建字符串对象都会和这个字典比较,没有就创建，重复了就用指针进行引用就可以了。相当于python对于字符串也是采用了对象池原理。(但是注意：如果字符串（含有空格），不可修改，没开启intern机制，不共用对象。比如"a b"和"a b",这种情况使用is不成立的形式 只有在命令行中可以。使用pycharm同样是True，因为做了优化)
总结：所以在python中如果创建了多个变量(不同变量名，此外不是通过变量引用方式创建的变量)，那么这些变量的引用地址都是不一样的。那么这些变量之间使用is 去比较的话，就是False的结果。但是除了小整数和字符串除外。如果是通过引用的方式创建的变量的话，那么可以参考
变量引用在内存中的复制存储原理博客：python变量的引用以及在底层存储原理

```
>>> ls =[1,2,3]  ###list类型 值虽然一样，但是两个变量，内存中分配了两个地址。
>>> ls1 =[1,2,3]
>>> ls == ls1
True
>>> ls is ls1
False
 
>>> t1 = (1,2,3) ##元组类型 值虽然一样，但是两个变量，内存中分配了两个地址。
>>> t2 = (1,2,3)
>>> t1 == t2
True
>>> t1 is t2
False
 
>>> d1 ={"1":2,"3":4}  ##dict类型 值虽然一样，但是两个变量，内存中分配了两个地址。
>>> d2 ={"1":2,"3":4}
>>> d1 == d2
True
>>> d1 is d2
False
>>> id(d1),id(d2)
(5425744, 4537872)
 
>>> st1 ='abc' #字符串类型 注意这里st1 和st2,值一样，内存地址也一样。因为字符串的intern机制。
>>> st2 ='abc'
>>> st1 == st2
True
>>> st1 is st2
True
 
>>> f1 = 3.14 #数值类型 值虽然一样，但是两个变量，内存中分配了两个地址。
>>> f2 = 3.14
>>> f1 == f2
True
>>> f1 is f2
False
 
>>> a = 1000 #超出[-5,256]范围值虽然一样，但是两个变量，内存中分配了两个地址。
>>> b = 1000
>>> a is b ,a ==b
(False, True)
 
>>> a = 1  #值在小整数对象池范围内，所以值一样，内存地址一样。
>>> b = 1
>>> a is b ,a ==b
(True, True)
>>>
```

5.python中对于None值的比较：使用is

```
>>> a =""
>>> a is None
False
>>> b ="aa"
>>> b is None
False
>>> b is not None
True
```

## 6、广播机制（Broadcasting in Python）

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/685f36c96c86fef53c3abc8fe509949c.png)

这是一个不同食物(每100g)中不同营养成分的卡路里含量表格，表格为3行4列，列表示不同的食物种类，从左至右依次为苹果，牛肉，鸡蛋，土豆。行表示不同的营养成分，从上到下依次为碳水化合物，蛋白质，脂肪。

那么，我们现在想要计算不同食物中不同营养成分中的卡路里百分比。

现在计算苹果中的碳水化合物卡路里百分比含量，首先计算苹果（100g）中三种营养成分卡路里总和56+1.2+1.8
= 59，然后用56/59 = 94.9%算出结果。

可以看出苹果中的卡路里大部分来自于碳水化合物，而牛肉则不同。

对于其他食物，计算方法类似。首先，按列求和，计算每种食物中（100g）三种营养成分总和，然后分别用不用营养成分的卡路里数量除以总和，计算百分比。

那么，能否不使用**for**循环完成这样的一个计算过程呢？

假设上图的表格是一个4行3列的矩阵$A$，记为 $A_{3\times 4}$，接下来我们要使用**Python**的**numpy**库完成这样的计算。我们打算使用两行代码完成，第一行代码对每一列进行求和，第二行代码分别计算每种食物每种营养成分的百分比。

在**jupyter notebook**中输入如下代码，按**shift+Enter**运行，输出如下。

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/56f38d09498335ae1155d2102f9b435d.png)

下面使用如下代码计算每列的和，可以看到输出是每种食物(100g)的卡路里总和。

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/c74557c22e724b1aa84ceeb3e5b6685d.png)

其中`sum`的参数`axis=0`表示求和运算按列执行，之后会详细解释。

接下来计算百分比，这条指令将 $3\times 4$的矩阵$A$除以一个$1 \times 4$的矩阵，得到了一个 $3 \times 4$的结果矩阵，这个结果矩阵就是我们要求的百分比含量。

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/aa852c608c711a73cb4d834a2956d9ae.png)

下面再来解释一下`A.sum(axis = 0)`中的参数`axis`。**axis用来指明将要进行的运算是沿着哪个轴执行，在numpy中，0轴是垂直的，也就是列，而1轴是水平的，也就是行。**

而第二个`A/cal.reshape(1,4)`指令则调用了**numpy**中的广播机制。这里使用 $3 \times 4$的矩阵$A$除以 $1 \times 4$的矩阵$cal$。技术上来讲，其实并不需要再将矩阵$cal$ `reshape`(重塑)成 $1 \times 4$，因为矩阵$cal$本身已经是 $1 \times 4$了。但是当我们写代码时不确定矩阵维度的时候，通常会对矩阵进行重塑来确保得到我们想要的列向量或行向量。重塑操作`reshape`是一个常量时间的操作，时间复杂度是$O(1)$，它的调用代价极低。

那么一个 $3 \times 4$ 的矩阵是怎么和 $1 \times 4$的矩阵做除法的呢？让我们来看一些更多的广播的例子。

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/537ee092a262cbdb874fe6e775039823.png)

在numpy中，当一个 $4 \times 1$的列向量与一个常数做加法时，实际上会将常数扩展为一个 $4 \times 1$的列向量，然后两者做逐元素加法。结果就是右边的这个向量。这种广播机制对于行向量和列向量均可以使用。

再看下一个例子。

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/d03358cec4b5ea22eb5dbef06fb6a1b7.png)

用一个 $2 \times 3$的矩阵和一个 $1 \times 3$ 的矩阵相加，其泛化形式是 $m \times n$ 的矩阵和 $1 \times n$的矩阵相加。在执行加法操作时，其实是将 $1 \times n$ 的矩阵复制成为 $m \times n$ 的矩阵，然后两者做逐元素加法得到结果。针对这个具体例子，相当于在矩阵的第一列加100，第二列加200，第三列加300。这就是在前一张幻灯片中计算卡路里百分比的广播机制，只不过这里是除法操作（广播机制与执行的运算种类无关）。

下面是最后一个例子

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/0fd16b22ad9b31d531ad7aa50f95cbbc.png)

这里相当于是一个 $m \times n$ 的矩阵加上一个 $m \times 1$ 的矩阵。在进行运算时，会先将 $m \times 1$ 矩阵水平复制 $n$ 次，变成一个 $m \times n$ 的矩阵，然后再执行逐元素加法。

广播机制的一般原则如下：

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/bdc1b1f6f0ba18659f140e47b26bf38b.png)

这里我先说一下我本人对**numpy**广播机制的理解，再解释上面这张PPT。

首先是**numpy**广播机制

**如果两个数组的后缘维度的轴长度相符或其中一方的轴长度为1，则认为它们是广播兼容的。广播会在缺失维度和轴长度为1的维度上进行。**

后缘维度的轴长度：`A.shape[-1]` 即矩阵维度元组中的最后一个位置的值

对于视频中卡路里计算的例子，矩阵 $A_{3,4}$ 后缘维度的轴长度是4，而矩阵 $cal_{1,4}$ 的后缘维度也是4，则他们满足后缘维度轴长度相符，可以进行广播。广播会在轴长度为1的维度进行，轴长度为1的维度对应`axis=0`，即垂直方向，矩阵 $$\text{cal}_{1,4}$$ 沿`axis=0`(垂直方向)复制成为 $$\text{cal_temp}_{3,4}$$ ，之后两者进行逐元素除法运算。

现在解释上图中的例子

矩阵 $A_{m,n}$ 和矩阵 $B_{1,n}$ 进行四则运算，后缘维度轴长度相符，可以广播，广播沿着轴长度为1的轴进行，即 $B_{1,n}$ 广播成为 ${B_{m,n}}'$ ，之后做逐元素四则运算。

矩阵 $A_{m,n}$ 和矩阵 $B_{m,1}$ 进行四则运算，后缘维度轴长度不相符，但其中一方轴长度为1，可以广播，广播沿着轴长度为1的轴进行，即 $B_{m,1}$ 广播成为 ${B_{m,n}}'$ ，之后做逐元素四则运算。

矩阵 $A_{m,1}$ 和常数$ R$ 进行四则运算，后缘维度轴长度不相符，但其中一方轴长度为1，可以广播，广播沿着缺失维度和轴长度为1的轴进行，缺失维度就是`axis=0`,轴长度为1的轴是`axis=1`，即$R$广播成为 ${B_{m,1}}'$ ，之后做逐元素四则运算。

最后，对于**Matlab/Octave** 有类似功能的函数`bsxfun`。

总结一下`broadcasting`，可以看看下面的图：

![](D:/%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99/deeplearning_ai_books-master/images/695618c70fd2922182dc89dca8eb83cc.png)



# 2、os使用

## 1、os.path.join(dirpath, name)

把目录和文件名合成一个路径，需要得到全路径,需要使用 os.path.join(dirpath, name)

## 2、os.walk()

```
import os
path = 'F:/source_files/python/'
for fpathe,dirs,fs in os.walk(path):
    for f in fs:
        print(os.path.join(fpathe,f))
```

返回的是一个三元tupple(dirpath, dirnames, filenames),其中第一个为起始路径，第二个为起始路径下的文件夹,第三个是起始路径下的文件。dirpath是一个string，代表目录的路径，dirnames是一个list，包含了dirpath下所有子目录的名字，filenames是一个list，包含了非目录文件的名字，这些名字不包含路径信息。

## 3、os.listdir()

os.listdir()这个函数，只会返回路径下的所有文件名（而这些文件名不含路径）

## 4、example

### 1、用python遍历一个图片文件夹，并输出所有路径到一个txt

```
# _*_ coding:utf8 _*_
import os
def listfiles(rootDir,txtfile,label=0):
    ftxtfile = open(txtfile,'w')
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:
        for f in files:
            srcname = os.path.join(root,f)
            srcformat = srcname.split('.')[-1]
            if srcformat == 'jpeg': 
                ftxtfile.write(srcname+' '+str(label)+'\n')
```

```
listfiles('./中华田园犬/','txtfile')
```

### 2、定义一个类，这个类将用于批量处理图像，包括若干功能：

(1)统一图像的格式为jpg。

(2)将多个图片文件夹构造成一个用于图像分类的数据集，产生txt文件，每一行的格式为：图片路径 标签

(3)txt文件按照7:3的比例分为训练集和测试集，并随机打乱顺序

```
# _*_ coding:utf8 _*_
import os
import sys
import shutil
import cv2
import random
class GeneDataset():
    def __init__(self, rootdir):
        self.rootdir = rootdir
        self.subdirs = []
        self.subdirimages = []
        self.numclasses = 0
        self.lines = []
        
    def looksubdir(self):
        list_dirs = os.walk(self.rootdir)
        for root, dirs, files in list_dirs:
            for d in dirs:
                self.subdirs.append(os.path.join(root,d))
                self.numclasses = self.numclasses + 1
    def reformat(self):
        label = 0
        for subdir in self.subdirs:
            list_dirs = os.walk(subdir)
            for root, dirs, files in list_dirs:
                for f in files:
                    srcname = os.path.join(root,f)
                    srcformat = srcname.split('.')[-1]
                    if srcformat is not ".jpg":
                        newname = srcname.replace(srcformat,'jpg')
                        self.lines.append(newname+' '+str(label)+'\n')
            label = label + 1 

    def split_train_val(self,trainfile,testfile):
        if len(self.lines):
            random.shuffle(self.lines)
            ftrainfile = open(trainfile,'w')
            ftestfile = open(testfile,'w')
            trainlength = int(0.7*len(self.lines))
            for i in range(0,trainlength):
                ftrainfile.write(self.lines[i])
            for i in range(trainlength,len(self.lines)):
                ftestfile.write(self.lines[i])
```

