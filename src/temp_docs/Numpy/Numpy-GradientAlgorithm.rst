.. _header-n0:

梯度下降算法(GD)
==================

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

.. _header-n3:

随机梯度下降算法(SGD)
---------------------

.. code:: python

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

.. _header-n5:

带动量的梯度下降算法(Momentum GD)
---------------------------------

.. code:: python

   def momentum_gradient_descent(parameters, grads, v, beta, learning_rate):
       L = len(parameters) // 2
       for l in range(L):
           v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
           v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
           parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
           parametes["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
       return parameters, v

.. _header-n7:

Adam 梯度下降算法
-----------------

.. code:: python

   def adam_gradient_descent(parameters, grads, v, s, t, learning_rate = 0.01, 
                             beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
       L = len(parameters) // 2
       v_corrected = {}                        
       s_corrected = {}                         
       # Perform Adam update on all parameters
       for l in range(L):
           v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads['dW'+str(l+1)]
           v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads['db'+str(l+1)]        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".   
           v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
           v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
           s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)])**2
           s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)])**2


           # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
           s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
           s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".

           parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
           parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)    
       return parameters, v, s

.. _header-n10:

RMSprop 梯度下降算法
--------------------

.. code:: python

   def RMSprop_gradient_descent():
       pass

.. _header-n13:

Adadelta 梯度下降算法
---------------------

.. code:: python

   def adadelta_gradient_descent():
       pass
