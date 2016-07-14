import tensorflow as tf
from rnnlhc.fitting.utilities import parse_data
from rnnlhc.fitting import BatchData
import argparse
import time
import numpy as np
'''
Class  rnnlhc is introduced as a model
This script also has a front facing main function to run the class.
Mayur Mudigonda, June 2016

'''
class RNNLHC:
    """ The RNN LHC Model with just simple number of hidden weights """
    def __init__(self,is_training,config):
       self.batch_size = batch_size = config.batch_size
       #self.num_steps = num_steps = config.num_steps
       #Reset Default Graph
       tf.reset_default_graph()

       #None for batch size, Max Num Steps, 3
       self.input_data = tf.placeholder(tf.float32, [None, config.MaxNumSteps, 3])
       self.targets = tf.placeholder(tf.float32, [None, config.MaxNumSteps,3])
       #self.input_data = tf.placeholder(tf.float32, [None, config.MaxNumSteps])
       #self.targets = tf.placeholder(tf.float32, [None, config.MaxNumSteps])
       #self.input_data = tf.placeholder(tf.float32, [None, None, 3])
       #self.targets = tf.placeholder(tf.float32, [None, None,3])

       #Weights and biases
       w = tf.Variable(tf.random_normal([config.hidden_size,config.feat_dims],stddev=0.1),trainable=True,name="matrix_w")
       b = tf.Variable(tf.zeros([config.feat_dims]),trainable=True,name="vector_b")

       sLen = tf.placeholder(tf.int32)
       loss = tf.Variable(0.,trainable = False)

       #istate = state = tf.placeholder(tf.float32,[None,config.hidden_size])
       x = tf.transpose(self.input_data,[1,0,2])
       x = tf.reshape(x,[-1,3])
       #We'll transform the data to a slightly higher dimension here
       '''
       x_higher = tf.matmul(x,w) + b
       x_split = tf.split(0,config.MaxNumSteps,x_higher)
       '''
       x_split = tf.split(0,config.MaxNumSteps,x)
       y = tf.transpose(self.targets,[1,0,2])
       y = tf.reshape(y,[-1,3])
       y_split = tf.split(0,config.MaxNumSteps,y)
       #x_split = tf.split(0,config.MaxNumSteps,self.input_data)
       #y_split = tf.split(0,config.MaxNumSteps,self.targets)


       #Initialize LSTM Basic Cell
       lstm = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,state_is_tuple=True)
       import IPython; IPython.embed()
       #lstm_multi = tf.nn.rnn_cell.MultiRNNCell([lstm]*config.num_layers,state_is_tuple=True)
       ### One could potentially expand this to multiple layers thusly
       # cell = tf.nn.rnn_cell.MultiRNNCell([lstm]*config.num_layers)
       #RNN
       #ops, states = tf.nn.rnn(lstm,x_split,dtype=tf.float32,sequence_length=sLen)
       ops, states = tf.nn.rnn(lstm,x_split,dtype=tf.float32)
       #ops, states = tf.nn.rnn(lstm_multi,x_split,dtype=tf.float32,sequence_length=sLen)
       #self._initial_state = lstm.zero_state(batch_size, tf.float32)
       #unclear if this should be a list
       for inp,target in zip(ops,y_split):
           op = tf.nn.relu(tf.matmul(inp,w)+b) 
           loss += self.loss_function(target,op)

       loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
       self._loss = loss
       self._sLen = sLen
       #optimizer
       #optimizer = tf.train.GradientDescentOptimizer(0.1)
       #self.train_op = optimizer.minimize(loss)
       self._lr = tf.Variable(0.0, trainable=False)
       tvars = tf.trainable_variables()
       grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      config.max_grad_norm)
       #optimizer = tf.train.GradientDescentOptimizer(self.lr)
       optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.5)
       self.train_op = optimizer.apply_gradients(zip(grads, tvars))
       self.states = states[-1]
       with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter('logs/',graph=sess.graph)
            for ii in range(np.int32(1e+3)):
                lr_decay = config.lr_decay ** max(0 - config.max_epoch, 0.0)
                self.assign_lr(sess,config.learning_rate*lr_decay)
                import IPython; IPython.embed()
                train_data,seq_len = config.BD.sample_batch(rand_int=rand_int,batch_size=config.batch_size)
                loss_op,train_op = sess.run([loss,self.train_op],{self.input_data:train_data[:,:-1,:],
                    self.targets:train_data[:,1:,:]})
                print("loss is {}, train_op".format(loss_op,train_op))

       return

    def loss_function(self,targets,inputs):
       loss = tf.reduce_mean((targets -inputs)**2)
       return loss

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


    @property
    def lr(self):
        return self._lr
    
    @property
    def states(self):
        return self.states

    @property
    def loss(self):
        return self._loss

    @property
    def sLen(self):
        return self._sLen 


class Config(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 16
  MaxNumSteps = 24
  feat_dims = 3
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 3 #If I understand this correctly, vocab is x,y,z 

class TestConfig(object):
  """Tiny config, for testing."""
  learning_rate = 1
  max_grad_norm = 0.1
  num_layers = 2
  MaxNumSteps = 15
  feat_dims = 1 
  hidden_size = 500
  max_epoch = 1
  lr_decay = 0.
  batch_size = 20

def get_config(config):
    if config == 0:
        return TestConfig()
    else:
        return Config()


def run_model(sess,m,data,eval_op,SeqLength,verbose=True):
  cost,_,state = sess.run([m.loss,eval_op,m.states],
                  {m.input_data: data[:,:-1,:],
                   m.targets: data[:,1:,:],
                   m.sLen: SeqLength})
  return cost,state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser to provide flags for the LSTM')
    parser.add_argument('--checkpoint',type=str,default="/media/Gondor/Data/rnnlhc/")
    parser.add_argument('--loaddata',type=str,default="../data/EventDump_10Ktracks.json")
    parser.add_argument('--config',type=int,default=0,help="1 Train, 0 Test")
    rand_int = 16
    args = parser.parse_args()
    json_data = parse_data(fname=args.loaddata)
    BD = BatchData.BatchData(json_data)
    config = get_config(args.config)
    config.BD = BD
    eval_config = get_config(args.config)
    eval_config.batch_size = 20 
    #eval_config.num_steps = 1

    print("Initializing train object")
    m = RNNLHC(is_training=True, config=config)
    import IPython; IPython.embed()
    print("Train object initialized")
    with tf.Session() as sess:
        print("Starting a session")
        sess.run(tf.initialize_all_variables())
        for ii in range(int(1e+4)):
            lr_decay = config.lr_decay ** max(0 - config.max_epoch, 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print("Learning rate is {}".format(config.learning_rate *lr_decay))
            print("Sample data")
            train_data,seq_len = BD.sample_batch(rand_int=rand_int,batch_size=config.batch_size)
            train_perplexity,state = run_model(sess,m,np.array(train_data),m.train_op,rand_int-1,verbose=True)
            print("Train Perplexity is {}".format(train_perplexity))
