# 1、init.py

1.文件组织架构 

 ├── checkpoints/ 
 ├── data/ 
 │   ├── **init**.py 
 │   ├── dataset.py 
 │   └── get_data.sh 
 ├── models/ 
 │   ├── **init**.py 
 │   ├── AlexNet.py 
 │   ├── BasicModule.py 
 │   └── ResNet34.py 
 └── utils/ 
 │   ├── **init**.py 
 │   └── visualize.py 
 ├── config.py 
 ├── main.py 
 ├── requirements.txt 
 ├── README.md 
 可以看到，几乎每个文件夹下都有**init**.py，一个目录如果包含了**init**.py 文件，那么它就变成了一个包（package）。**init**.py可以为空，也可以定义包的属性和方法，**但其必须存在，其它程序才能从这个目录中导入相应的模块或函数**。 
 例如在data/文件夹下有**init**.py，则在main.py 中就可以 
 from data.dataset import DogCat

而如果在data/**init**.py中写入 
 from .dataset import DogCat 
 则在main.py中就可以直接写为： 
 from data import DogCat

 

或者 
 import data; 
 dataset = data.DogCat 
 相比于from data.dataset import DogCat更加便捷。

# 2、ImageFolder

ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字。如果数据集已经按类别分好例如train文件夹下又包含dog和cat文件夹则可以用ImageFolder函数

```
train/dog/xxx.png
train/dog/xxy.png
train/dog/xxz.png
 
train/cat/123.png
train/cat/nsdf3.png
train/cat/asd932_.png
```

```
ImageFolder(root,transform=None,target_transform=None,loader=default_loader)
```

root : 在指定的root路径下面寻找图片 
transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象 
target_transform :对label进行变换 
loader: 指定加载图片的函数，默认操作是读取PIL image对象

```
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

#加上transforms
normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.RandomResizedCrop (224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])

dataset=ImageFolder('data/dogcat_2/',transform=transform)
```

```
#dataloader是一个可迭代的对象，意味着我们可以像使用迭代器一样使用它 或者 or batch_datas, batch_labels in dataloader:
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)

dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size()) # batch_size, channel, height, weight
#输出 torch.Size([3, 3, 224, 224])
```

# 3、自定义数据集

```
#数据处理
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

transform=transforms.Compose([
    transforms.Resize(224), #缩放图片，保持长宽比不变，最短边的长为224像素,
    transforms.CenterCrop(224), #从中间切出 224*224的图片
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) #标准化至[-1,1]
])

#定义自己的数据集合
class DogCat(data.Dataset):

    def __init__(self,root,transform):
        #所有图片的绝对路径
        imgs=os.listdir(root)

        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path=self.imgs[index]
        #dog-> 1 cat ->0
        label=1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img=Image.open(img_path)
        if self.transforms:
            data=self.transforms(pil_img)
        else:
            pil_img=np.asarray(pil_img)
            data=torch.from_numpy(pil_img)
        return data,label

    def __len__(self):
        return len(self.imgs)

dataSet=DogCat('./data/dogcat',transform=transform)

print(dataSet[0])

```

