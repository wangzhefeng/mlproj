.. _header-n0:

Numpy-DNN
=========

步骤：

-  定义网络结构

   -  指定输入层、隐藏层、输出层的大小

-  初始化模型参数

-  循环操作：执行前向传播 => 计算当前损失 => 执行反向传播 => 权值更新

   -  执行前向传播

   -  计算当前损失

   -  执行反向传播

   -  权值更新

.. _header-n23:

1.定义网络结构(指定输入层、隐藏层、输出层的大小)
------------------------------------------------

Input layer:

:math:`H_{(hidden^{[1]}, 1)}^{[1]} = \sigma^{[1]}\big(W_{(hidden^{[1]}, input)}^{[1]} \times X_{(input, 1)} + b_{(hidden^{[1]}, 1)}^{[1]}\big)`

Hidden layers:

:math:`H_{(hidden^{[2]}, 1)}^{[2]} = \sigma^{[2]}\big(W_{(hidden^{[2]}, hidden^{[1]})}^{[2]} \times H_{(hidden^{[1]}, 1)}^{[1]} + b_{(hidden^{[2]}, 1)}^{[2]}\big)`

:math:`\vdots`

:math:`H_{(hidden^{[i]}, 1)}^{[i]} = \sigma^{[i]}\big(W_{(hidden^{[i]}, hidden^{[i - 1]})}^{[i]} \times H_{(hidden^{[i-1]}, 1)}^{[i-1]} + b_{(hidden^{[i]}, 1)}^{[i]}\big)`

:math:`\vdots`

:math:`H_{(hidden^{[L - 1]}, 1)}^{[L -1]} = \sigma^{[L-1]}\big(W_{(hidden^{[L - 1]}, hidden^{[L - 2]})}^{[L-1]} \times H_{(hidden^{[L-2]}, 1)}^{[L-2]} + b_{(hidden^{[L-1]}, 1)}^{[L-1]}\big)`

Output layer:

:math:`Y = \sigma^{[L]}\big(W_{(hidden^{[L]}, hidden^{[L - 1]})}^{[L]} \times H_{(hidden^{[L-1]}, 1)}^{[L-1]} + b_{(hidden^{[L]}, 1)}^{[L]}\big)`

Data:

+--------+--------+--------+--------+--------+--------+--------+--------+
| :math: | :math: | :math: | :math: | :math: | :math: | :math: | :math: |
| `x_1`  | `x_2`  | `x_3`  | `x_4`  | `x_5`  | `y_1`  | `y_2`  | `y_3`  |
+========+========+========+========+========+========+========+========+
|        |        |        |        |        |        |        |        |
+--------+--------+--------+--------+--------+--------+--------+--------+
|        |        |        |        |        |        |        |        |
+--------+--------+--------+--------+--------+--------+--------+--------+

.. code:: python

   input = 5
   hidden = 4
   output = 3
   layer_dims = [input, hidden, output]

.. _header-n66:

2.初始化模型参数
----------------

.. code:: python

   import numpy as np

   def initialize_parameters_deep(layer_dims):
       """
       Arguments:
           layer_dims: 神经网络各层维数, list
       Returns:
           parameters: {"params": value_array}
       """
       np.random.seed(3)
       parameters = {}

       # 网路层数: input + hidden + output
       L = len(layer_dims)

       # l = 1, 2, ..., L - 1
       for l in range(1, L): 
           # W.shape = (layer_dims[l], layer_dims[l - 1])
           parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01 
           # b.shape = (layer_dims[l], 1)
           parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
       assert(parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
       assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))

       return parameters

.. _header-n68:

3.循环操作
----------

前向传播、后向传播计算流程图：

.. image:: ../../images/backforward.png
   :alt: 

.. _header-n71:

3.1 前向传播
~~~~~~~~~~~~

定义激活函数:

:math:`y = \frac{1}{1 + e^{-x}}`

.. code:: python

   def sigmoid(x):
       y = 1 / (1 + np.exp(-x))
       return y

:math:`y = max \\{0, x\\}`

.. code:: python

   def relu(z):
       y = np.maximum(0, x)
       return y

前向传播:

   -  Input: :math:`(x^{(i)}, y^{(i)})`

   -  Hidden:

      -  :math:`z^{[1]\(i\)} = W^{[1]} x^{(i)} + b^{[1]\(i\)}`

      -  :math:`a^{[1]\(i\)} = relu(z^{[1]\(i\)})`

      -  :math:`z^{[2]\(i\)} = W^{[2]} a^{[1]\(i\)} + b^{[2]\(i\)}`

   -  Output:
      :math:`\hat{y}^{(i)} = a^{[2]\(i\)}  = \sigma(z^{[2]\(i\)})`

.. code:: python

   def linear_forward(A_prev, W, b):
       Z = np.dot(W, A_prev) + b
       linear_cache = {
           "Z": z
       }
       return Z, linear_cache

   def linear_activation_forward(A_prev, W, b, activation):
       """
       Arguments:
           A_prew: 前一步执行前向计算的结果
       """
       if activation == "sigmoid":
           Z, linear_cache = linear_forward(A_prev, W, b)
           A, activation_cache = sigmoid(Z)
       elif activation == "relu":
           Z, linear_cache = linear_forward(A_prev, W, b)
           A, activation_cache = relu(Z)

       assert (A.shape == (W.shape[0], A_prew.shape[1]))
       cache = (linear_cache, activation_cache)
       return A, cache

   def L_model_forward(x, parameters):
       caches = []
       A = x
       # 神经网络的层数
       L = len(parameters) // 2
       # linear -> relu
       for l in ragne(1, L):
           A_prev = A
           A, cache = linear_activation_forward(A_prev, 
                                                W = parameters["W" + str(l)], 
                                                b = parameters["b" + str(l)], 
                                                activation = "relu")
           caches.append(cache)
       # Linear -> SIGMOID
       AL, cache = linear_activation_forward(A, 
                                             W = parameters["W" + str(L)], 
                                             b = parameters["b" + str(L)], 
                                             activation = "sigmoid")
       caches.append(cache)
       assert (AL.shape == (1, x.shape[1]))

       return AL, caches

.. _header-n94:

3.2 计算前向损失
~~~~~~~~~~~~~~~~

:math:`J = -\frac{1}{m}\sum_{i=0}^{m}\Big(y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\Big)`

.. code:: python

   def compute_cost(AL, y):
       m = y.shape[1]
       J = np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL))
       cost = - np.sum(J) / m
       assert (cost.shape == ())
       return cost

.. _header-n97:

3.3 后向传播
~~~~~~~~~~~~

   -  :math:`LOSS = L(y^{(i)} - \hat{y}^{(i)}) \\\\
          = L(a^{[2]\(i\)} - \hat{y}^{(i)}) \\\\
          = L(\sigma(z^{[2]\(i\)}) - \hat{y}^{(i)}) \\\\
          = L(\sigma(W^{2} \times a^{[1]\(i\)} + b^{[2]\(i\)}) - \hat{y}^{(i)})\\\\
          = L(\sigma(W^{2} \times relu(z^{[1]\(i\)}) + b^{[2]\(i\)}) - \hat{y}^{(i)}) \\\\
          = L(\sigma(W^{2} \times relu(W^{1} \times x^{(i)} + b^{[1]\(i\)}) + b^{[2]\(i\)}) - \hat{y}^{(i)})`

      -  :math:`\frac{\partial L}{\partial L} = 1`

      -  :math:`\frac{\partial L}{\partial a^{[2]\(i\)}} = \frac{\partial L}{\partial L} \times \frac{\partial L}{\partial a^{[2]\(i\)}}`

      -  :math:`\frac{\partial L}{\partial z^{[2]\(i\)}} = \frac{\partial L}{\partial L} \times \frac{\partial L}{\partial a^{[2]\(i\)}} \times \frac{\partial a^{[2]\(i\)}}{\partial z^{[2]\(i\)}}`

      -  :math:`\frac{\partial L}{\partial a^{[1]\(i\)}} = \frac{\partial L}{\partial L} \times \frac{\partial L}{\partial a^{[2]\(i\)}} \times \frac{\partial a^{[2]\(i\)}}{\partial z^{[2]\(i\)}} \times \frac{\partial z^{[2]\(i\)}}{\partial a^{[1]\(i\)}}`

      -  :math:`\frac{\partial L}{\partial z^{[1]\(i\)}} = \frac{\partial L}{\partial L} \times \frac{\partial L}{\partial a^{[2]\(i\)}} \times \frac{\partial a^{[2]\(i\)}}{\partial z^{[2]\(i\)}} \times \frac{\partial z^{[2]\(i\)}}{\partial a^{[1]\(i\)}} \times \frac{\partial a^{[1]\(i\)}}{\partial z^{[1]\(i\)}}`

.. code:: python

   def linear_activation_backward(dZ, cache):
       A_prev, W, b = cache
       m = A_prev.shape[1]

       dW = np.dot(dZ, A_prev.T) / m
       db = np.sum(dZ, axis = 1, keepdims = True) / m
       dA_prev = np.dot(W.T, dZ)

       assert (dA_prev.shape == A_prev)
       assert (dW.shape == W.shape)
       assert (db.shape == b.shape)
       return dA_prev, dW, db

   def L_model_backward(AL, Y, caches):
       grads = {}
       L = len(caches) 
       # the number of layers
       m = AL.shape[1]
       Y = Y.reshape(AL.shape) 
       # after this line, Y is the same shape as AL

       # Initializing the backpropagation
       dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    
       # Lth layer (SIGMOID -> LINEAR) gradients
       current_cache = caches[L-1]
       grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")    
       for l in reversed(range(L - 1)):
           current_cache = caches[l]
           dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
           grads["dA" + str(l + 1)] = dA_prev_temp
           grads["dW" + str(l + 1)] = dW_temp
           grads["db" + str(l + 1)] = db_temp    
       return grads

更新参数:

.. code:: python

   def gradient_descent(parameters, grads, learning_rate):
       """
       # 梯度下降法(Gradient Descent)
       Arguments:
           parameters: python dict containing parameters to be updated.
           grads: python dict containing gradients to update each parameters.
           learning_rate: the learning rate, scalar.
       Returns:
           parameters: python dict contain updated parameters
       """
       # number of layers in the neural networks 
       L = len(parameters) // 2
       for l in range(L):
           parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
           parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
       return parameters 


   def random_mini_batch(x, y, batch_size = 64, seed = 0):
       np.random.seed(seed)
       m = x.shape[1]
       # setp 1: shuffle (x, y)
       mini_batches = []
       permutation = list(np.random.permutation(m))
       shuffled_x = x[:, permutation]
       shuffled_y = y[:, permutation].reshape((1, m))
       # step 2: partition (shuffled_x, shuffled_y)
       num_complete_minibatches = math.floor(m / batch_size)
       for k in range(0, num_complete_minibatches):
           mini_batch_x = shuffled_x[:, 0:batch_size]
           mini_batch_y = shuffled_y[:, 0:batch_size]
           mini_batch = (mini_batch_x, mini_batch_y)
           mini_batches.append(mini_batch)
       if m % batch_size != 0:
           mini_batch_x = shuffled_x[:, 0:m - batch_size * math.floor(m / batch_size)]
           mini_batch_y = shuffled_y[:, 0:m - batch_size * math.floor(m / batch_size)]
           mini_batch = (mini_batch_x, mini_batch_y)
           mini_batches.append(mini_batch)

       return mini_batches



   def stochastic_gradient_descent(parameters, grads, learning_rate):
       pass


   def momentum_gradient_descent(parameters, grads, v, beta, learning_rate):
       L = len(parameters) // 2
       for l in range(L):
           v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
           v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
           parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
           parametes["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
       return parameters, v

模型实现：

.. code:: python

   def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
       np.random.seed(1)
       costs = []    

       # Parameters initialization.
       parameters = initialize_parameters_deep(layers_dims)    
       # Loop (gradient descent)
       for i in range(0, num_iterations):        
           # Forward propagation: 
           # [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
           AL, caches = L_model_forward(X, parameters)        
           # Compute cost.
           cost = compute_cost(AL, Y)        
           # Backward propagation.
           grads = L_model_backward(AL, Y, caches)        
           # Update parameters.
           parameters = gradient_descent(parameters, grads, learning_rate)        
           # Print the cost every 100 training example
           if print_cost and i % 100 == 0:            
               print ("Cost after iteration %i: %f" %(i, cost))        if print_cost and i % 100 == 0:
               costs.append(cost)    
       # plot the cost
       plt.plot(np.squeeze(costs))
       plt.ylabel('cost')
       plt.xlabel('iterations (per tens)')
       plt.title("Learning rate =" + str(learning_rate))
       plt.show()    
       
       return parameters

正则化 L1, L2：

.. code:: python

   def compute_cost_with_regularization(A3, Y, parameters, lambd):    
       """
       Implement the cost function with L2 regularization.
       Arguments:
           A3: post-activation, output of forward propagation, of shape (output size, number of examples)
           Y: "true" labels vector, of shape (output size, number of examples)
           parameters: python dictionary containing parameters of the model
       Returns:
           cost: value of the regularized loss function (formula (2))
       """
       m = Y.shape[1]
       W1 = parameters["W1"]
       W2 = parameters["W2"]
       W3 = parameters["W3"]
       cross_entropy_cost = compute_cost(A3, Y)
       L2_regularization_cost = 1 / m * lambd / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
       cost = cross_entropy_cost + L2_regularization_cost    
       return cost

   def backward_propagation_with_regularization(X, Y, cache, lambd):    
       """
       Implements the backward propagation of baseline model to which added an L2 regularization.
       Arguments:
           X: input dataset, of shape (input size, number of examples)
           Y: "true" labels vector, of shape (output size, number of examples)
           cache: cache output from forward_propagation()
           lambd: regularization hyperparameter, scalar
       Returns:
           gradients: A dict with the gradients with respect to each parameter, 
                      activation and pre-activation variables
       """
       m = X.shape[1]
       (Z1, A1, W1, b1,
        Z2, A2, W2, b2, 
        Z3, A3, W3, b3) = cache
       dZ3 = A3 - Y
       dW3 = 1./m * np.dot(dZ3, A2.T) +  lambd/m * W3
       db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
       dA2 = np.dot(W3.T, dZ3)
       dZ2 = np.multiply(dA2, np.int64(A2 > 0))
       dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * W2
       db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
       dA1 = np.dot(W2.T, dZ2)
       dZ1 = np.multiply(dA1, np.int64(A1 > 0))
       dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m * W1
       db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
       gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,                 
                    "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}    
       return gradients

正则化 Dropout:

.. code:: python

   def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
       np.random.seed(1)
       W1 = parameters["W1"]
       b1 = parameters["b1"]
       W2 = parameters["W2"]
       b2 = parameters["b2"]
       W3 = parameters["W3"]
       b3 = parameters["b3"]
       # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
       Z1 = np.dot(W1, X) + b1
       A1 = relu(Z1)
       D1 = np.random.rand(A1.shape[0], A1.shape[1])    
       D1 = D1 < keep_prob                             
       A1 = np.multiply(D1, A1)                         
       A1 = A1 / keep_prob                             
       Z2 = np.dot(W2, A1) + b2
       A2 = relu(Z2)
       D2 = np.random.rand(A2.shape[0], A2.shape[1])     
       D2 = D2 < keep_prob                             
       A2 = np.multiply(D2, A2)                       
       A2 = A2 / keep_prob                           
       Z3 = np.dot(W3, A2) + b3
       A3 = sigmoid(Z3)
       cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)    
       return A3, cache

   def backward_propagation_with_dropout(X, Y, cache, keep_prob):
       m = X.shape[1]
       (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
       dZ3 = A3 - Y
       dW3 = 1./m * np.dot(dZ3, A2.T)
       db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
       dA2 = np.dot(W3.T, dZ3)
       dA2 = np.multiply(dA2, D2)   
       dA2 = dA2 / keep_prob        
       dZ2 = np.multiply(dA2, np.int64(A2 > 0))
       dW2 = 1./m * np.dot(dZ2, A1.T)
       db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
       dA1 = np.dot(W2.T, dZ2)
       dA1 = np.multiply(dA1, D1)   
       dA1 = dA1 / keep_prob           
       dZ1 = np.multiply(dA1, np.int64(A1 > 0))
       dW1 = 1./m * np.dot(dZ1, X.T)
       db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
       gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,                 
                    "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}    
       return gradients
