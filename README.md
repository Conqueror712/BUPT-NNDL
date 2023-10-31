## 零、快速开始

1. 克隆项目并进入根目录

```
git clone git@github.com:Conqueror712/Cifar-10-Project.git
cd Cifar-10-Project
```

2. 安装所需的第三方库

```
pip install -r requirements.txt
```

3. 运行代码

```
python main.py
```


## 一、实验目的

本次实验的目的是设计和实现一个基于卷积神经网络的图像分类模型，理解 CNN 基本结构、训练过程，研究正则化方法和交叉验证在改进模型泛化能力和选择超参数中的作用。

## 二、方法

1. CNN 网络结构

    选择ResNet18作为主干网络,它在`get_net()`函数中使用`d2l.resnet18()`实现。

2. 数据预处理

    这里使用了将原始图像使用`torchvision`中的转换方法进行格式转换和标准化；使用`transforms.Resize()`对图像大小进行预处理以及通过`Normalize()`标准化每个通道等等方法。

3. 正则化方法

    代码中，在卷积层之间加入`BatchNorm()`层，实现特征标准化。

4. 交叉验证

    将 train 数据集随机划分为 train 和 validation 子集，在`train()`函数中使用`valid_iter`反复评估验证集效果。

5. 优化方法

    使用`SGD()`优化器训练模型，加入`StepLR()`学习率衰减策略，并且使用 GPU 加速训练。

6. 预测和评估

    在`train()`和`get_net()`函数中完成模型预测，在`test_iter`上计算预测结果。

> 流程简单概括如下：

首先是加载数据集，其次我们使用图像增广来解决过拟合的问题。在训练中，我们进行了随机水平翻转图像、对彩色图像的三个 RGB 通道执行标准化等操作。在测试期间，我们只对图像执行标准化，以消除评估结果中的随机性。

然后，我们读取由原始图像组成的数据集，每个样本都包括一张图片和一个标签。在训练期间，我们指定了上面定义的所有图像增广操作。 当验证集在超参数调整过程中用于模型评估时，不应引入图像增广的随机性。 在最终预测之前，我们根据训练集和验证集组合而成的训练模型进行训练，以充分利用所有标记的数据。

再之后是定义模型，我们使用 ResNet-18 模型。以及定义训练函数，将根据模型在验证集上的表现来选择模型并调整超参数，然后我们定义了模型训练函数。

最后是训练和验证模型，在获得具有超参数的满意的模型后，我们使用所有标记的数据来重新训练模型并对测试集进行分类。

## 三、数据集

数据集分为训练集和测试集，共涵盖 10 个类别：飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车。

## 四、实验结果

> 以下是训练和验证结果：

<img src="/img/01.png" alt="image" style="zoom:67%;" />

> 以下是测试的结果（请忽略未响应）：

<img src="/img/02.png" alt="image" style="zoom:67%;" />

## 五、回答问题

1. CNN 是常用于图像处理任务的一种神经网络结构，其基本结构包括卷积层、池化层和全连接层等。
2. 对于模型的泛化能力，Dropout 和多种 Normalization 方法可以起到一定的正则化作用，有助于减少过拟合和提高模型的泛化能力。
3. 通过交叉验证，可以为神经网络找到最好的 Hyperparameters，是因为它将数据集划分为多个互斥的子集，然后依次使用其中一部分作为验证集，其余部分作为训练集进行多次训练和验证。在每一次训练和验证中，可以尝试不同的超参数组合，比较它们在验证集上的性能，从而找到最佳的超参数组合。

---

FIN.
