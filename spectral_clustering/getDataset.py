import random
from sklearn import datasets

 
def genTwoCircles(n_samples=1000):
    '''
     datasets.make_circles在2d中创建一个包含较小圆的大圆的样本集
     factor内圈和外圈之间的比例因子
     noise高斯噪声的标准偏差加到数据上
     返回：
     X生成的样本
     y每个样本的类成员的整数标签（0或1）
    '''
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y


def genMoons(n_samples=1000):
    X, y = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=0.05, random_state=None)
    return X, y


def genBlocks(products_num=1000,max_block=10,blocks_num=100):
    ''' generate random blocks '''
    pids = range(1, products_num+1)
    blocks = {}
    for bid in range(1,blocks_num+1):
        block_len = random.randint(1, max_block)
        block = random.sample(pids, block_len)
        blocks[bid] = block
    return blocks




