import torch




w = torch.tensor([1.0], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

def loss_func(pred_y, train_y):
    """
    损失函数为loss = (y1 - y) ^ 2 = (w * x - y) ^ 2
    求导后为 ∂loss/∂w = 2(w * x - y) * x
    """
    return (pred_y - train_y) ** 2


train_x1, train_x2 = torch.tensor([1.0]), torch.tensor([2.0])
train_y1, train_y2 = torch.tensor([2.0]), torch.tensor([4.0])


pred_y1 = w * train_x1
loss1 = loss_func(pred_y1, train_y1)
loss1.backward()

"""
w=1, train_x1=1, pred_y1=1
∂loss/∂w=2*(1-2)*1=-2
所以w的梯度为-2
"""

print(f"第一次推导之后w的梯度为{w.grad}")

pred_y2 = w * train_x2
loss2 = loss_func(pred_y2, train_y2)
loss2.backward()

"""
w=1, train_x2=2, pred_y1=4
∂loss/∂w=2*(2-4)*2=-8
所以w的梯度为-2
"""
print(f"第二次推导之后w的梯度为{w.grad}")

