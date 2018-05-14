import numpy as np

# define the number of iterations.
num_itr = 1000

# define batch size.
batchSize = 3

# define the input data dimension.
inputSize = 2

# define the output dimension.
outputSize = 1

# define the dimension of the hidden layer.
hiddenSize = 3


class Neural_Network():
    def __init__(self):     
        #weights
        self.U = np.random.randn(inputSize, hiddenSize) 
        self.W = np.random.randn(hiddenSize, outputSize) 
        self.e = np.random.randn(hiddenSize) 
        self.f = np.random.randn(outputSize) 


    def fully_connected(self, X, U, e):
        '''
        fully connected layer.
        inputs:
            U: weight 
            e: bias
        outputs:
            X * U + e
        '''
        return np.dot(X, U) + e


    def sigmoid(self, s):
        '''
        sigmoid activation function. 
        inputs: s
        outputs: sigmoid(s)  
        '''
        return 1/(1+np.exp(-s))


    def sigmoidPrime(self, s):
        '''
        derivative of sigmoid (Written section, Part a).
        inputs: 
            s = sigmoid(x)
        outputs: 
            derivative sigmoid(x) as a function of s 
        '''
        d_sigmoid = s*(1-s)
        return d_sigmoid


    def forward(self, X):
        '''
        forward propagation through the network.
        inputs:
            X: input data (batchSize, inputSize) 
        outputs:
            c: output (batchSize, outputSize)
        '''
        self.z = self.fully_connected(X, self.U, self.e)
        self.b = self.sigmoid(self.z)
        self.h = self.fully_connected(self.b, self.W, self.f)
        c = self.sigmoid(self.h)

        return c


    def d_loss_o(self, gt, o):
        '''
        computes the derivative of the L2 loss with respect to 
        the network's output.
        inputs:
            gt: ground-truth (batchSize, outputSize)
            o: network output (batchSize, outputSize)
        outputs:
            d_o: derivative of the L2 loss with respect to the network's 
            output o. (batchSize, outputSize)
        '''
        d_o =(o - gt)/ batchSize

        return d_o


    def error_at_layer2(self, d_o, o):
        '''
        computes the derivative of the loss with respect to layer2's output
        (Written section, Part b).
        inputs:
            d_o: derivative of the loss with respect to the network output (batchSize, outputSize)
            o: the network output (batchSize, outputSize)
        returns 
            delta_k: the derivative of the loss with respect to the output of the second
            fully connected layer (batchSize, outputSize).
        '''
        delta_k = d_o * self.sigmoidPrime(o)
        return delta_k


    def error_at_layer1(self, delta_k, W, b):
        '''
        computes the derivative of the loss with respect to layer1's output (Written section, Part e).
        inputs:
            delta_k: derivative of the loss with respect to the output of the second
            fully connected layer (batchSize, outputSize). 
            W: the weights of the second fully connected layer (hiddenSize, outputSize).
            b: the input to the second fully connected layer (batchSize, hiddenSize).
        returns:
            delta_j: the derivative of the loss with respect to the output of the second
            fully connected layer (batchSize, hiddenSize).
        '''
        delta_j = delta_k.dot(W.T) * self.sigmoidPrime(b)
        return delta_j


    def derivative_of_w(self, b, delta_k):
        '''
        computes the derivative of the loss with respect to W (Written section, Part c).
        inputs:
            b: the input to the second fully connected layer (batchSize, hiddenSize).
            delta_k: the derivative of the loss with respect to the output of the second
            fully connected layer's output (batchSize, outputSize).
        returns:
            d_w: the derivative of loss with respect to W  (hiddenSize ,outputSize).
        '''
        d_w = b.T.dot(delta_k)
        return d_w


    def derivative_of_u(self, X, delta_j):
        '''
        computes the derivative of the loss with respect to U (Written section, Part f).
        inputs:
            X: the input to the network (batchSize, inputSize).
            delta_j: the derivative of the loss with respect to the output of the first
            fully connected layer's output (batchSize, hiddenSize).
        returns:
            d_u: the derivative of loss with respect to U (inputSize, hiddenSize).
        '''
        d_u = X.T.dot(delta_j)
        return d_u


    def derivative_of_e(self, delta_j):
        '''
        computes the derivative of the loss with respect to e (Written section, Part g).
        inputs:
            delta_j: the derivative of the loss with respect to the output of the first
            fully connected layer's output (batchSize, hiddenSize).
        returns:
            d_e: the derivative of loss with respect to e (hiddenSize).
        '''
        d_e = np.sum(delta_j, axis=0)
        return d_e


    def derivative_of_f(self, delta_k):
        '''
        computes the derivative of the loss with respect to f (Written section, Part d).
        inputs:
            delta_k: the derivative of the loss with respect to the output of the second
            fully connected layer's output (batchSize, outputSize).
        returns:
            d_f: the derivative of loss with respect to f (outputSize).
        '''
        d_f = np.sum(delta_k, axis=0)
        return d_f


    def backward(self, X, gt, o):
        '''
        backpropagation through the network.
        Task: perform the 8 steps required below.
        inputs: 
            X: input data (batchSize, inputSize)
            y: ground truth (batchSize, outputSize)
            o: network output (batchSize, outputSize)        
        '''

        # 1. Compute the derivative of the loss with respect to c.
        # Call: d_loss_o
        d_loss_o = self.d_loss_o(gt, o)

        # 2. Compute the error at the second layer (Written section, Part b).
        # Call: error_at_layer2
        c = self.sigmoid(self.h)
        error_at_layer2 = self.error_at_layer2(d_loss_o, c)

        # 3. Compute the derivative of W (Written section, Part c).
        # Call: derivative_of_w
        derivative_of_w = self.derivative_of_w(self.b , error_at_layer2)

        # 4. Compute the derivative of f (Written section, Part d).
        # Call: derivative_of_f
        derivative_of_f = self.derivative_of_f(error_at_layer2)

        # 5. Compute the error at the first layer (Written section, Part e).
        # Call: error_at_layer1 

        error_at_layer1 = self.error_at_layer1(error_at_layer2, self.W, self.b)

        # 6. Compute the derivative of U (Written section, Part f).
        # Call: derivative_of_u
        derivative_of_u = self.derivative_of_u(X, error_at_layer1)

        # 7. Compute the derivative of e (Written section, Part g).
        # Call: derivative_of_e
        derivative_of_e = self.derivative_of_e(error_at_layer1)

        # 8. Update the parameters
        learning_rate = 1
        batch_Size = X.shape[0]
        self.W += -learning_rate * derivative_of_w
        self.f += -learning_rate * derivative_of_f
        self.U += -learning_rate * derivative_of_u
        self.e += -learning_rate * derivative_of_e

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
      

def main():
    """ Main function """
    # generate random input data of dimension (batchSize, inputSize). 
    a = np.random.randint(0, high=10, size=[batchSize,2], dtype='l')

    # generate random ground truth.
    t = np.random.randint(0, high=100, size=[batchSize,1], dtype='l')

    # scale the input and output data.
    a = a/np.amax(a, axis=0) 
    t = t/100 

    # create an instance of Neural_Network.
    NN = Neural_Network()
    for i in range(num_itr): 
        if i % 50 == 0:
            print("Input: \n" + str(a)) 
            print("Actual Output: \n" + str(t))
            print("Predicted Output: \n" + str(NN.forward(a)))
            print("Loss: \n" + str(np.mean(np.square(t - NN.forward(a)))))
            print("\n")
        NN.train(a, t)


if __name__ == "__main__":
    main()