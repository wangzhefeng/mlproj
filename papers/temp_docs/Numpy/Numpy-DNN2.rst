.. _header-n0:

Numpy-DNN2
==========

深度神经网络：

-  定义网络结构(指定输入层、隐藏层、输出层的大小)

   -  :math:`H_{4 \times 1} = \sigma^{(1)}\Big(W_{4 \times 5}^{(1)} \times X_{5 \times 1} + b_{5 \times 1}^{(1)}\Big) \\\\
      H_{3 \times 1} = \sigma^{(2)}\Big(W_{3 \times 4}^{(2)} \times H_{4 \times 1} + b_{3 \times 1}^{(2)}\Big)`

-  初始化模型参数

   -  ``initialize_parameters_deep()``

-  循环操作：执行前向传播 => 计算损失 => 执行后向传播 => 权值更新

   -  前向传播

      -  前向传播的基本过程就是执行加权线性计算和对线性计算的结果进行激活函数处理的过程

   -  损失函数

   -  后向传播

   -  权重更新

前向传播、后向传播计算流程图：

.. image:: ../../images/backforward.png
   :alt: 

前向传播:

   -  INPUT = :math:`(x^{(i)}, y^{(i)})`

   -  :math:`z^{[1]\(i\)} = W^{1} x^{(i)} + b^{[1]\(i\)}`

   -  :math:`a^{[1]\(i\)} = relu(z^{[1]\(i\)})`

   -  :math:`z^{[2]\(i\)} = W^{2} a^{[1]\(i\)} + b^{[2]\(i\)}`

   -  :math:`\hat{y}^{(i)} = a^{[2]\(i\)}  = \sigma(z^{[2]\(i\)})`

前向损失：

   -  LOSS = :math:`L(y^{(i)} - \hat{y}^{(i)})`

后向传播:

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

   import numpy as np

   # 初始化模型参数
   def initialize_parameters_deep(layer_dims):
   	np.random.seed(3)
   	parameters = {}
   	L = len(layer_dims)
   	for l in range(1, L):
   		parameters["W" + str(l)] = np.random.randn(layer_dims[1], layer_dims[l - 1]) * 0.01
   		parameters["b" + str(l)] = np.zeros((layer_dims[1], 1))

   	assert(parameters["W" + str(l)].shape == (layer_dims[1], layer_dims[l - 1]))
   	assert(parameters["b" + str(len)].shape == (layer_dims[1], 1))
   	return parameters





   # 前向传播
   def sigmoid(x):
   	"""
   	sigmoid function
   	"""
   	y = 1 / (1 + np.exp(-x))
   	return y

   def relu(z):
   	pass

   def linear_forward(A_predv, W, b):
   	pass

   def linear_activation_forward(A_prev, W, b, activation):
   	"""
   	Arguments:
   		A_prew: 前一步执行前向计算的结果
   		W:
   		b:
   		activation:
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
   	L = len(parameters) // 2
   	for l in ragne(1, L):
   		A_prev = A
   		A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
   		cache.append(cache)
   	AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
   	cache.append(cache)
   	assert (AL.shape == (1, x.shape[1]))
   	return AL, cache

   # 计算前向损失
   def compute_cost(AL, y):
   	m = y.shape[1]
   	J = np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL))
   	cost = -np.sum(J) / m
   	assert (cost.shape == ())
   	return cost


   # 后向传播
   def linear_backward(dZ, cache):
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

   def update_parameters(parameters, grads, learning_rate):
       # number of layers in the neural network
       L = len(parameters) // 2 
       # Update rule for each parameter. Use a for loop.
       for l in range(L):
           parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
           parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]    
       return parameters

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
           parameters = update_parameters(parameters, grads, learning_rate)        
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

   def compute_cost_with_regularization(A3, Y, parameters, lambd):    """
       Implement the cost function with L2 regularization. See formula (2) above.

       Arguments:
       A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
       Y -- "true" labels vector, of shape (output size, number of examples)
       parameters -- python dictionary containing parameters of the model

       Returns:
       cost - value of the regularized loss function (formula (2))
       """
       m = Y.shape[1]
       W1 = parameters["W1"]
       W2 = parameters["W2"]
       W3 = parameters["W3"]

       cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost


       L2_regularization_cost = 1/m * lambd/2 * (np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))

       cost = cross_entropy_cost + L2_regularization_cost    
       return cost

   def backward_propagation_with_regularization(X, Y, cache, lambd):    """
       Implements the backward propagation of our baseline model to which we added an L2 regularization.

       Arguments:
       X -- input dataset, of shape (input size, number of examples)
       Y -- "true" labels vector, of shape (output size, number of examples)
       cache -- cache output from forward_propagation()
       lambd -- regularization hyperparameter, scalar

       Returns:
       gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
       """

       m = X.shape[1]
       (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

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

       gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}    
       return gradients

正则化 Dropout:

.. code:: python

   def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
       np.random.seed(1)    # retrieve parameters
       W1 = parameters["W1"]
       b1 = parameters["b1"]
       W2 = parameters["W2"]
       b2 = parameters["b2"]
       W3 = parameters["W3"]
       b3 = parameters["b3"]    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
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

       gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}    
       return gradients
