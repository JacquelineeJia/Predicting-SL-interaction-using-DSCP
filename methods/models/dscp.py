import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorly as tl
tl.set_backend('tensorflow')


# This function is the kth order matricization of the tensor X
@tf.function
def unfold(X,k):
    X = tf.expand_dims(X,axis = 0) 

    s = []
    for i in range(3):
      s.append(i+1)
   
    s.pop(k)
    s.insert(0,k+1)

    X = tfkl.Permute(s)(X)
    X = tf.reshape(X,[tf.shape(X)[1],-1])
    return X


# This function randomly initizes the parameter matrices P
def P_rand_init(shape,rank):
  
    P = []
    P.append(2*(np.random.uniform(size = (shape[0],rank))-.5))
    P.append(2*(np.random.uniform(size = (shape[2],rank))-.5))

    return P
   
# This function normalizes the parameter matrices P
def normalize(P):
  
  P_norm = []
  lamb = tf.ones(P[0].shape[-1])
  for p in P:
    p_norm,norm = tf.linalg.normalize(p,axis = 0)
    lamb = lamb * norm
    P_norm.append(p_norm)

  return P_norm

# This function solves for the nth factor matrix of X given the parameter matrics P
def LeastSquares(X,P,V,n):

    X_n = unfold(X,n)
  
    P_norm = normalize(P)
    tl.set_backend('tensorflow')
    print(tl.get_backend())
    P_neg = tl.tenalg.khatri_rao(P_norm,skip_matrix = n)

    V_skip = V[:n] + V[n+1:]
    V_ = V_skip[0]
    for v in V_skip[1:]:
      V_ = V_*v

    V_p = tf.linalg.pinv(V_)

    C_n = tf.einsum('bi,ij,jr->br',X_n,P_neg,V_p)

    return C_n


# This class defines Selective CANDCOMP/PARAFAC
class SCP(tfkl.Layer):

  def __init__(self,X,rank,shape):
    super(SCP,self).__init__()

    self.P_true = P_rand_init(shape=shape,rank=rank)
    self.X = tf.constant(X,dtype = 'float32')

  def build(self,input_shape):
   
    self.P = []
    
    relevent = []
    relevent.append(tf.reshape(self.P_true[0],-1))
    relevent.append(tf.reshape(self.P_true[1],-1))
     
    for p in self.P_true:
     self.P.append(tf.Variable(
        initial_value = tf.cast(p,'float32'),
        trainable = True,
        shape = p.shape,dtype = 'float32'))

    
    setattr(self,"P0", self.P[0])
    setattr(self,"P2", self.P[1])


  def call(self,inputs):

    C = []
    V = []

    # Get V, V is just a computation shortcut and has no meaning or use except to calculate the factor matricse C
    for p in self.P[:]:
      V.append(tf.matmul(p,p,transpose_a = True))

    # Get the factor matrices C
    for i in range(3): 
      out = LeastSquares(self.X,[self.P[0],self.P[0],self.P[1]],V,i)
      out = tf.gather(out,inputs[i],axis = 0)
      C.append(out)
    
    return C

# This function attaches an MLP to SCP, resulting in Deep Selective CANDCOMP/PARAFAC
class DSCP(tfk.Model):

  def __init__(self,X,classifier,shape,rank,nn,hidden_do,out_do,l1_strength, l2_strength,activation_function):
    super(DSCP, self).__init__()
 
    self.classifier = classifier(rank,nn,hidden_do,out_do,l1_strength, l2_strength,activation_function) 
    self.scp = SCP(X=X,shape=shape,rank=rank) 

  def call(self,dataset):

    i = 0
    inputs = []
    for i in range(3):
      inputs.append(tf.cast(dataset[:,i],dtype = 'int32'))

    C = self.scp(inputs)

    da = C[0]
    db = C[1]
    cl = C[2]
    print("gene shape:", da.shape)
    print("gene shape:", db.shape)
    print("cancer shape:", cl.shape) 
    out = self.classifier([da,db,cl])
 
    return out 
