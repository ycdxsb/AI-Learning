import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
 
dtype = torch.FloatTensor
 
sentences = ['i like cat', 'i love coffee', 'i hate milk']
#将上面的单词逐个分开
word_list = " ".join(sentences).split()
#将分词后的结果去重
word_list = list(set(word_list))
#对单词建立索引，for循环里面是先取索引，再取单词
word_dict = {w:i for i, w in enumerate(word_list)}
#反向建立索引
number_dict = {i:w for i, w in enumerate(word_list)}
#计算词典长度
n_class = len(word_dict)
 
#NNLM的计算步长
n_step = len(sentences[0].split())-1
#隐藏层的参数量
n_hidden = 2
#嵌入词向量的维度
m = 2
 
#构建输入输出数据
def make_batch(sentences):
    input_batch = []
    target_batch = []
 
    for sen in sentences:
        word = sen.split()#将句子中每个词分词
        #:-1表示取每个句子里面的前两个单词作为输入
        #然后通过word_dict取出这两个单词的下标，作为整个网络的输入
        input = [word_dict[n] for n in word[:-1]] # [0, 1], [0, 3], [0, 5]
        #target取的是预测单词的下标，这里就是cat,coffee和milk
        target = word_dict[word[-1]] # 2, 4, 6
        
        #输入数据集
        input_batch.append(input) # [[0, 1], [0, 3], [0, 5]]
        #输出数据集
        target_batch.append(target) # [2, 4, 6]
 
    return input_batch, target_batch
 
input_batch, target_batch = make_batch(sentences)
#将数据装载到torch上
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)
 
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
 
#定义网络结构，继承nn.Module
class NNLM(nn.Module):
  def __init__(self):
    super(NNLM, self).__init__()
    #计算词向量表，大小是len(word_dict) * m
    self.C = nn.Embedding(n_class, m)
    #下面就是初始化网络参数，公式如下
    """
    hiddenout = tanh(d + X*H)
    y = b + X*H + hiddenout*U
    """
    self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
    self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
    self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
    self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
    self.b = nn.Parameter(torch.randn(n_class).type(dtype))
 
  def forward(self, X):
    '''
    X: [batch_size, n_step]
    '''
    #根据词向量表，将我们的输入数据转换成三维数据
    #将每个单词替换成相应的词向量
    X = self.C(X) # [batch_size, n_step] => [batch_size, n_step, m]
    #将替换后的词向量表的相同行进行拼接
    #view的第一个参数为-1表示自动判断需要合并成几行
    X = X.view(-1, n_step * m) # [batch_size, n_step * m]
    hidden_out = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
    output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U) # [batch_size, n_class]
    return output
 
model = NNLM()
#分类问题用交叉熵作为损失函数
criterion = nn.CrossEntropyLoss()
#优化器使用Adam
#所谓的优化器，实际上就是你用什么方法去更新网路中的参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
 
# 开始训练
for epoch in range(5000):
  for batch_x, batch_y in loader:
    optimizer.zero_grad()
    output = model(batch_x)
 
    # output : [batch_size, n_class], batch_y : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, batch_y)
    #每1000次打印一次结果
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
 
    #反向传播更新梯度
    loss.backward()
    optimizer.step()
 
# Predict
# max()取的是最内层维度中最大的那个数的值和索引，[1]表示取索引
predict = model(input_batch).data.max(1, keepdim=True)[1]
 
# Test
# squeeze()表示将数组中维度为1的维度去掉
print([sen.split()[:n_step] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])