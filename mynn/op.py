from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """

        self.input = X

        assert self.input.shape[1] == self.params['W'].shape[0], "The input size doesn't match the weights size."

        output = np.matmul(X, self.params['W']) + self.params['b']
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.matmul(self.input.T, grad) # [input, batch] * [batch, output]
        self.grads['b'] = np.matmul(np.ones(grad.shape[0]), grad).reshape(1, -1) # [batch] * [batch, out]

        output =  np.matmul(grad, self.params['W'].T)

        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class GlobalAvePool(Layer):
    """
    Input: [C, H, W]
    output: [C]
    """
    def __init__(self):
        super().__init__()
        self.H = None
        self.W = None
        self.optimizable =False
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        _, self.C, self.H, self.W = X.shape
        out = X.reshape(_, self.C, -1)
        out = np.mean(out, axis=-1)
        return out

    def backward(self, grads):
        assert grads.shape[-1] == self.C
        out = grads.reshape(-1, self.C, 1, 1)
        out = np.tile(out, (1, 1, self.H, self.W))
        out /= self.H * self.W
        return out


class conv2D(Layer):
    """
    The 2D convolutional layer. For now with no padding and no stride.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(1, out_channels, in_channels, kernel_size, kernel_size)).astype('float64')
        self.b = initialize_method(size=(1, out_channels, 1, 1)).astype('float64')
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}
        self.input = None

        # shape info
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.start = int(kernel_size / 2)

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """

        self.input = X.astype('float64')

        self.batch_size = X.shape[0]
        self.new_H = int((X.shape[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.new_W = int((X.shape[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1
        output = np.zeros((self.batch_size, self.out_channels, self.new_H, self.new_W))
 
        new_h = 0
        for h in range(self.start, X.shape[2] - self.start, self.stride):
            new_W = 0
            for w in range(self.start, X.shape[3] - self.start, self.stride):
                temp_X = X[:, :, h-self.start: h+self.start+1, w-self.start: w+self.start+1].reshape(self.batch_size, 1, -1, self.params['W'].shape[-2], self.params['W'].shape[-1])
                temp = (temp_X * self.params['W']).reshape(self.batch_size, self.out_channels, -1)
                out = np.sum(temp, axis=-1) # out shape : [batch_size, out_channels]
                output[:, :, new_h, new_W] = out
                new_W += 1
            new_h += 1
    
        return output + self.params['b']

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        temp = np.sum(grads, axis=0, keepdims=True)
        temp = np.sum(temp, axis=2, keepdims=True)
        self.grads['b'] = np.sum(temp, axis=3, keepdims=True)


        self.grads['W'] = np.zeros_like(self.params['W'])
        output = np.zeros_like(self.input)

        new_h = 0
        for h in range(self.start, self.input.shape[2] - self.start, self.stride):
            new_W = 0
            for w in range(self.start, self.input.shape[3] - self.start, self.stride):
                target_grads = grads[:, :, new_h, new_W].reshape(self.batch_size, -1, 1, 1, 1)
                #W's grad
                # temp_X's shape [batch, 1, in, k, k]
                temp_X = self.input[:, :, h-self.start: h+self.start+1, w-self.start: w+self.start+1].reshape(self.batch_size, 1, -1, self.params['W'].shape[-2], self.params['W'].shape[-1])
                # temp's shape [batch, out, in, k, k]
                temp = temp_X * target_grads
                self.grads['W'] += np.sum(temp, axis=0, keepdims=True)

                # input's grad
                # temp's shape [batch, in, k, k]
                temp = target_grads * self.params['W']
                temp = np.sum(temp, axis=1)
                output[:, :, h-self.start: h+self.start+1, w-self.start: w+self.start+1] += temp

                new_W += 1
            new_h += 1
        
        return output
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.predicts = None 
        self.eps = 1e-15
        self.labels = None
        self.num = None
        self.grads = None
        self.model = model
        self.max_classes = max_classes

        self.has_softmax = True

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        """
        assert labels.max() < predicts.shape[-1]
        self.predicts = predicts
        self.labels = labels.astype('int32')
        self.num = self.predicts.shape[0]

        if self.has_softmax:
            self.outputs = softmax(predicts)
        else:
            self.outputs = predicts
        
        batch_index = np.arange(self.num)
        loss = -np.sum(np.log(self.outputs[batch_index, labels] + self.eps))

        return loss / self.num
    
    def backward(self):

        one_hot_labels =  np.eye(self.max_classes)[self.labels]

        if self.has_softmax:
            self.grads = -(1 / self.num) * (one_hot_labels - self.outputs)
        else:
            self.grads = -(1 / self.num) * self.predicts

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    

class L2Regularization(Layer):
    pass

        
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition