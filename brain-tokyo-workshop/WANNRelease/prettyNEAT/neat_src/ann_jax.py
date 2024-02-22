import jax.numpy as jnp
from jax.ops import index, index_update

def getNodeOrder(nodeG, connG):
    """Builds connection matrix from genome through topological sorting, adapted for JAX.

    Args:
      nodeG - (jnp_array) - node genes
              [3 X nUniqueGenes]
              [0,:] == Node Id
              [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
              [2,:] == Activation function (as int)

      connG - (jnp_array) - connection genes
              [5 X nUniqueGenes] 
              [0,:] == Innovation Number (unique Id)
              [1,:] == Source Node Id
              [2,:] == Destination Node Id
              [3,:] == Weight Value
              [4,:] == Enabled?  

    Returns:
      Q    - [int]      - sorted node order as indices
      wMat - (jnp_array) - ordered weight matrix
             [N X N]

      OR

      False, False      - if cycle is found
    """
    conn = connG.at[3, connG[4,:] == 0].set(jnp.nan)  # Use jax.numpy and jax.ops for mutation
    node = nodeG
    nIns = jnp.sum(node[1,:] == 1) + jnp.sum(node[1,:] == 4)
    nOuts = jnp.sum(node[1,:] == 2)
    
    # Convert IDs to indices
    lookup = node[0,:].astype(int)
    src = jnp.array([jnp.where(lookup == s)[0][0] for s in conn[1,:]]).astype(int)
    dest = jnp.array([jnp.where(lookup == d)[0][0] for d in conn[2,:]]).astype(int)
    
    wMat = jnp.zeros((node.shape[1], node.shape[1]))
    wMat = wMat.at[src, dest].set(conn[3,:])
    connMat = wMat[nIns+nOuts:, nIns+nOuts:]
    connMat = jnp.where(connMat != 0, 1, 0)

    # Topological Sort of Hidden Nodes
    edge_in = jnp.sum(connMat, axis=0)
    Q = jnp.where(edge_in == 0)[0]
    for i in range(len(connMat)):
        if (len(Q) == 0) or (i >= len(Q)):
            return False, False  # Cycle found, can't sort
        edge_out = connMat[Q[i], :]
        edge_in = edge_in - edge_out
        nextNodes = jnp.setdiff1d(jnp.where(edge_in == 0)[0], Q)  # JAX has no direct setdiff1d, consider alternatives
        Q = jnp.hstack((Q, nextNodes))

        if jnp.sum(edge_in) == 0:
            break
    
    # Add In and outs back and reorder wMat according to sort
    Q = Q + nIns + nOuts
    Q = jnp.concatenate((lookup[:nIns], Q, lookup[nIns:nIns+nOuts]), axis=0)
    wMat = wMat[Q[:, None], Q]

    return Q, wMat


def getLayer(wMat):
  """Get layer of each node in weight matrix
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1. Input and output nodes are ignored and assigned layer
  0 and max(X)+1 at the end.

  Args:
    wMat  - (np_array) - ordered weight matrix
           [N X N]

  Returns:
    layer - [int]      - layer # of each node

  Todo:
    * With very large networks this might be a performance sink -- especially, 
    given that this happen in the serial part of the algorithm. There is
    probably a more clever way to do this given the adjacency matrix.
  """
  wMat[np.isnan(wMat)] = 0  
  wMat[wMat!=0]=1
  nNode = np.shape(wMat)[0]
  layer = np.zeros((nNode))
  while (True): # Loop until sorting is stable
    prevOrder = np.copy(layer)
    for curr in range(nNode):
      srcLayer=np.zeros((nNode))
      for src in range(nNode):
        srcLayer[src] = layer[src]*wMat[src,curr]   
      layer[curr] = np.max(srcLayer)+1    
    if all(prevOrder==layer):
      break
  return layer-1


# -- ANN Activation ------------------------------------------------------ -- #

def act(weights, aVec, nInput, nOutput, inPattern):
  """Returns FFANN output given a single input pattern
  If the variable weights is a vector it is turned into a square weight matrix.
  
  Allows the network to return the result of several samples at once if given a matrix instead of a vector of inputs:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)

  Args:
    weights   - (np_array) - ordered weight matrix or vector
                [N X N] or [N**2]
    aVec      - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
    nInput    - (int)      - number of input nodes
    nOutput   - (int)      - number of output nodes
    inPattern - (np_array) - input activation
                [1 X nInput] or [nSamples X nInput]

  Returns:
    output    - (np_array) - output activation
                [1 X nOutput] or [nSamples X nOutput]
  """
  # Turn weight vector into weight matrix
  if np.ndim(weights) < 2:
      nNodes = int(np.sqrt(np.shape(weights)[0]))
      wMat = np.reshape(weights, (nNodes, nNodes))
  else:
      nNodes = np.shape(weights)[0]
      wMat = weights
  wMat[np.isnan(wMat)]=0

  # Vectorize input
  if np.ndim(inPattern) > 1:
      nSamples = np.shape(inPattern)[0]
  else:
      nSamples = 1

  # Run input pattern through ANN    
  nodeAct  = np.zeros((nSamples,nNodes))
  nodeAct[:,0] = 1 # Bias activation
  nodeAct[:,1:nInput+1] = inPattern

  # Propagate signal through hidden to output nodes
  iNode = nInput+1
  for iNode in range(nInput+1,nNodes):
      rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
      nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct) 
      #print(nodeAct)
  output = nodeAct[:,-nOutput:]   
  return output

def applyAct(actId, x):
  """Returns value after an activation function is applied
  Lookup table to allow activations to be stored in numpy arrays

  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent [tanh] (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine
  case 11 -- Squared

  Args:
    actId   - (int)   - key to look up table
    x       - (???)   - value to be input into activation
              [? X ?] - any type or dimensionality

  Returns:
    output  - (float) - value after activation is applied
              [? X ?] - same dimensionality as input
  """
  if actId == 1:   # Linear
    value = x

  if actId == 2:   # Unsigned Step Function
    value = 1.0*(x>0.0)
    #value = (np.tanh(50*x/2.0) + 1.0)/2.0

  elif actId == 3: # Sin
    value = np.sin(np.pi*x) 

  elif actId == 4: # Gaussian with mean 0 and sigma 1
    value = np.exp(-np.multiply(x, x) / 2.0)

  elif actId == 5: # Hyperbolic Tangent (signed)
    value = np.tanh(x)     

  elif actId == 6: # Sigmoid (unsigned)
    value = (np.tanh(x/2.0) + 1.0)/2.0

  elif actId == 7: # Inverse
    value = -x

  elif actId == 8: # Absolute Value
    value = abs(x)   
    
  elif actId == 9: # Relu
    value = np.maximum(0, x)   

  elif actId == 10: # Cosine
    value = np.cos(np.pi*x)

  elif actId == 11: # Squared
    value = x**2
    
  else:
    value = x

  return value


# -- Action Selection ---------------------------------------------------- -- #

def selectAct(action, actSelect):  
  """Selects action based on vector of actions

    Single Action:
    - Hard: a single action is chosen based on the highest index
    - Prob: a single action is chosen probablistically with higher values
            more likely to be chosen

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned 

  Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

  Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
  """  
  if actSelect == 'softmax':
    action = softmax(action)
  elif actSelect == 'prob':
    action = weightedRandom(np.sum(action,axis=0))
  else:
    action = action.flatten()
  return action

def softmax(x):
    """Compute softmax values for each sets of scores in x.
    Assumes: [samples x dims]

    Args:
      x - (np_array) - unnormalized values
          [samples x dims]

    Returns:
      softmax - (np_array) - softmax normalized in dim 1
    
    Todo: Untangle all the transposes...    
    """    
    if x.ndim == 1:
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=0)
    else:
      e_x = np.exp(x.T - np.max(x,axis=1))
      return (e_x / e_x.sum(axis=0)).T

def weightedRandom(weights):
  """Returns random index, with each choices chance weighted
  Args:
    weights   - (np_array) - weighting of each choice
                [N X 1]

  Returns:
    i         - (int)      - chosen index
  """
  minVal = np.min(weights)
  weights = weights - minVal # handle negative vals
  cumVal = np.cumsum(weights)
  pick = np.random.uniform(0, cumVal[-1])
  for i in range(len(weights)):
    if cumVal[i] >= pick:
      return i
        

# -- File I/O ------------------------------------------------------------ -- #

def exportNet(filename,wMat, aVec):
  indMat = np.c_[wMat,aVec]
  np.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')

def importNet(fileName):
  ind = np.loadtxt(fileName, delimiter=',')
  wMat = ind[:,:-1]     # Weight Matrix
  aVec = ind[:,-1]      # Activation functions

  # Create weight key
  wVec = wMat.flatten()
  wVec[np.isnan(wVec)]=0
  wKey = np.where(wVec!=0)[0] 

  return wVec, aVec, wKey

