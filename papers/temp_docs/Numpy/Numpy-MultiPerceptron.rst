.. _header-n0:

Numpy-MultiPerceptron
=====================

步骤：

-  定义网络结构

   -  指定输入层、隐藏层、输出层的大小

-  初始化模型参数

-  循环操作：执行前向传播 => 计算当前损失 => 执行反向传播 => 权值更新

   -  执行前向传播

   -  计算当前损失

   -  执行反向传播

   -  权值更新

.. code:: python

   import numpy as np

.. _header-n24:

1.定义神经网络结构
------------------

.. image:: ../../images/one_layer_network.png
   :alt: 

$$data = [ \\begin{matrix} x\ *1, & x*\ 2, & y \\end{matrix} ] ^{T}

.. math::

   $$input = \Big[
   \begin{matrix}
      x_1 \\\\
      x_2
   \end{matrix}
   \Big]

:math:`output = y`

$$\begin{matrix} a\ *{1}^{[1]} \\\\ a*\ {2}^{[1]} \\\\ a\ *{3}^{[1]}
\\\\ a*\ {4}^{[1]} \\end{matrix} = tanh \\Bigg( \\begin{matrix}
W\ *{1,1}^{[1]} & W*\ {2,1}^{[1]} & W\ *{3,1}^{[1]} & W*\ {4,1}^{[1]}
\\\\ W\ *{1,2}^{[1]} & W*\ {2,2}^{[1]} & W\ *{3,2}^{[1]} &
W*\ {4,2}^{[1]} \\end{matrix} ^{T} \\times \\begin{matrix} x\ *1 \\\\
x*\ 2 \\\\ \\end{matrix} + \\begin{matrix} b\ *{1}^{[1]} \\\\
b*\ {2}^{[1]} \\\\ b\ *{3}^{[1]} \\\\ b*\ {4}^{[1]} \\end{matrix}
\\Bigg)

.. math:: 

y = sigmoid \\Bigg( \\begin{matrix} W\ *{1}^{[2]} & W*\ {2}^{[2]} &
W\ *{3}^{[2]} & W*\ {4}^{[2]} \\end{matrix} \\times \\begin{matrix}
a\ *1^{[1]} \\\\ a*\ 2^{[1]} \\\\ a\ *3^{[1]} \\\\ a*\ 4^{[1]}
\\end{matrix} + \\begin{matrix} b\ *{1}^{[2]} \\\\ b*\ {2}^{[2]} \\\\
b\ *{3}^{[2]} \\\\ b*\ {4}^{[2]} \\end{matrix} \\Bigg) $$

.. code:: python

   x = [x1, x2] # x.shape = (1, 2)
   y = y        # y.shape = ()

.. code:: python

   def layer_size(x, y):
   	n_x = x.shape[0]
   	n_h = 4
   	n_y = y.shape[0]
   	return (n_x, n_h, n_y)

.. _header-n36:

2.初始化模型参数
----------------

模型参数：

-  :math:`W_{(n_h = 4, n_x)}^{[1]}`

-  :math:`b_{(n_h = 4, 1)}^{[1]}`

-  :math:`W_{(n_y, n_h = 4)}^{[2]}`

-  :math:`b_{(n_y, 1)}^{[2]}`

.. code:: python

   def initialize_parameters(n_x, n_h, n_y):
   	W1 = np.random.randn(n_h, n_x) * 0.01
   	b1 = np.zeros((n_h, 1))
   	W2 = np.random.randn(n_y, n_h) * 0.01
   	b2 = np.zeros((n_y, 1))

   	assert(W1.shape == (n_h, n_x))
   	assert(b1.shape == (n_h, 1))
   	assert(W2.shape == (n_y, n_h))
   	assert(b2.shape == (n_y, 1))

   	parameters = {
   		"W1": W1,
   		"b1": b1,
   		"W2": W2,
   		"b2": b2
   	}
   	return parameters

.. _header-n48:

3.循环操作
----------

.. _header-n49:

定义激活函数
~~~~~~~~~~~~

:math:`sigmoid(x) = \frac{1}{1 + e^{-x}}`

.. code:: python

   def sigmoid(x):
       y = 1 / (1 + np.exp(-x))
       return y

:math:`tanh(x) = x`

.. code:: python

   def tanh(x):
       y = np.tanh(x)
       return y

.. _header-n54:

前向传播
~~~~~~~~

:math:`z^{[1]\(i\)} = W^{[1]\(i\)} x^{(i)} + b^{[1]\(i\)}`

:math:`a^{[1]\(i\)} = tanh(z^{[1]\(i\)})`

:math:`z^{[2]\(i\)} = W^{[2]\(i\)} a^{[1]\(i\)} + b^{[2]\(i\)}`

:math:`\hat{y}^{(i)} = a^{[2]\(i\)}  = \sigma(z^{[2]\(2\)})`

.. code:: python

   def forward_propagation(x, parameters):
   	W1 = parameters["W1"]
   	b1 = parameters["b1"]
   	W2 = parameters["W2"]
   	b2 = parameters["b2"]
   	z1 = np.dot(W1, x) + b1
   	a1 = np.tanh(z1)
   	z2 = np.dot(W2, a1) + b2
   	y_hat = a2 = sigmoid(z2)
   	assert(a2.shape == (1, x.shape[1]))
   	cache = {
   		"z1" = z1,
   		"a1" = a1,
   		"z2" = z2
   		"a2" = a2
   	}
   	return a2, cache

.. _header-n62:

定义损失函数
~~~~~~~~~~~~

:math:`J = -\frac{1}{m}\sum_{i=0}^{m}\Big(y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\Big)`

.. code:: python

   def loss_func(y, y_hat):
   	m = y.shape[1]
   	logprobs = np.multipy(np.log(y_hat), y) + np.multipy(np.log(1 - y_hat), 1 - y)
   	cost = -1 / m * np.sum(logprobs)
   	cost = np.squeeze(cost)
   	assert(isinstance(cost, float))
   	return cost

.. _header-n66:

反向传播
~~~~~~~~

:math:`dz^{[2]} = a^{[2]} - y = A^{[2]} - Y`

:math:`dW^{[2]} = dz^{[2]}a^{[1]T} = \frac{1}{m} \times dZ^{[2]}A^{[1]T}`

:math:`db^{[2]} = dz^{[2]} = \frac{1}{m} \times np.sum(dZ^{[2]}, axis = 1, keepdims = True)`

:math:`dz^{[1]} = W^{[2]T}dz^{[2]} \times g^{[1]'}(z^{[1]}) = W^{[2]}dZ^{[2]} \times g^{[1]'}(Z^{[1]})`

:math:`dW^{[1]} = dz^{[1]}x^{T} = \frac{1}{m} \times dZ^{[1]}X^{T}`

:math:`db^{[1]} = dz^{[1]} = \frac{1}{m} \times np.sum(dZ^{[1]}, axis = 1, keepdims = True)`

.. code:: python

   def backward_propagation(parameters, cache, X, Y):
       m = X.shape[1]  
       # First, retrieve W1 and W2 from the dictionary "parameters".
       W1 = parameters['W1']
       W2 = parameters['W2']    
       # Retrieve also A1 and A2 from dictionary "cache".
       A1 = cache['A1']
       A2 = cache['A2']    
       # Backward propagation: calculate dW1, db1, dW2, db2. 
       dZ2 = A2 - Y
       dW2 = 1/m * np.dot(dZ2, A1.T)
       db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
       dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
       dW1 = 1/m * np.dot(dZ1, X.T)
       db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)

       grads = {"dW1": dW1,
                "db1": db1,                      
                "dW2": dW2,             
                "db2": db2}   
       return grads

.. _header-n75:

权重更新
~~~~~~~~

:math:`W^{[1]} = W^{[1]} - \eta \times dW^{[1]}`

:math:`b^{[1]} = b^{[1]} - \eta \times db^{[1]}`

:math:`W^{[2]} = W^{[2]} - \eta \times dW^{[2]}`

:math:`b^{[2]} = b^{[2]} - \eta \times db^{[2]}`

.. code:: python

   def update_parameters(parameters, grads, learning_rate = 1.2):
       # Retrieve each parameter from the dictionary "parameters"
       W1 = parameters['W1']
       b1 = parameters['b1']
       W2 = parameters['W2']
       b2 = parameters['b2']    
       # Retrieve each gradient from the dictionary "grads"
       dW1 = grads['dW1']
       db1 = grads['db1']
       dW2 = grads['dW2']
       db2 = grads['db2']    
       # Update rule for each parameter
       W1 -= dW1 * learning_rate
       b1 -= db1 * learning_rate
       W2 -= dW2 * learning_rate
       b2 -= db2 * learning_rate

       parameters = {"W1": W1, 
                     "b1": b1,            
                     "W2": W2,   
                     "b2": b2}    
       return parameters

.. code:: python

   def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
       np.random.seed(3)
       n_x = layer_sizes(X, Y)[0]
       n_y = layer_sizes(X, Y)[2]    
       # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
       parameters = initialize_parameters(n_x, n_h, n_y)
       W1 = parameters['W1']
       b1 = parameters['b1']
       W2 = parameters['W2']
       b2 = parameters['b2']    
       # Loop (gradient descent)
       for i in range(0, num_iterations):        
       # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
           A2, cache = forward_propagation(X, parameters)        
           # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
           cost = loss_func(A2, Y)        
           # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
           grads = backward_propagation(parameters, cache, X, Y)        
           # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
           parameters = update_parameters(parameters, grads, learning_rate = 1.2)      
           # Print the cost every 1000 iterations
           if print_cost and i % 1000 == 0:          
               print ("Cost after iteration %i: %f" %(i, cost))    
          
       return parameters
