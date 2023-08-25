import numpy as np

class node:
    def __init__(self, value):
        self.value = value
    def backward(self):
        pass

features = node(np.array([
    [1, 2],
    [2,3],
    [1,5]
]))
labels = node(np.array([[-1, 1, 2]]).T)

class operation:
    def __init__(self, *args):
        for input_name, input_node in zip(self.input_names, args):
            setattr(self,input_name,input_node)
        self.value = self.forward()
    def backward(self):
        grad_tup = self.gradient()
        for input_name, grad in zip(self.input_names, grad_tup):
            input_node = getattr(self, input_name)
            input_node.grad = grad
            input_node.backward()

class mm(operation):
    input_names = ("left", "right")
    def forward(self):
        return np.matmul(self.left.value, self.right.value)
    def gradient(self):
        # left is  n x p
        # right is p x u
        # a is     n x u
        return (
            np.matmul(self.grad, self.right.value.T),
            np.matmul(self.left.value.T, self.grad))

class relu(operation):
    input_names = ("a_mat",)
    def forward(self):
        return np.where(self.a_mat.value < 0, 0, self.a_mat.value)
    def gradient(self):
        a_grad = np.where(self.a_mat.value < 0, 0, 1)
        return (self.grad * a_grad,)

class mean_square_loss(operation):
    input_names = ("pred_vec", "label_vec")
    def forward(self):
        self.diff_vec = self.pred_vec.value - self.label_vec.value
        sq_loss_vec = 0.5 * self.diff_vec ** 2
        return sq_loss_vec.mean()
    def gradient(self):
        return (self.diff_vec, -self.diff_vec)

weight_vec = node(np.random.normal(size=(features.value.shape[1], 1)))
weight_vec.grad
a = mm(features, weight_vec)
J = mean_square_loss(a, labels)
J.backward()
weight_vec.grad

units_per_layer = [features.value.shape[1], 10, 5, 1]
weight_mat_list = []
prev_features = features
for layer_i in range(1, len(units_per_layer)):
    print(layer_i)
    w_size = units_per_layer[layer_i-1], units_per_layer[layer_i]
    print(w_size)
    weight_mat = node(np.random.normal(size=w_size))
    weight_mat_list.append(weight_mat)
    a_mat = mm(prev_features, weight_mat)
    if layer_i == len(units_per_layer)-1:
        J = mean_square_loss(a_mat, labels)
    else:
        prev_features = relu(a_mat)    
J.backward()
[w.grad.shape for w in weight_mat_list]
