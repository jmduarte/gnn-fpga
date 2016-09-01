import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#SEQ_LENGTH = 10

class testrnn:
    def __init__(self,config):
        self.config = config
        tf.reset_default_graph()
        with tf.variable_scope("rnnlhc") as train_scope:
            self.input_data = tf.placeholder(tf.float32,[config.batch_size,config.MaxNumSteps])
            self.eval_input_data = tf.placeholder(tf.float32,[1,config.MaxNumSteps])#create eval node, 3 time steps
            #self.eval_target = tf.Variable(tf.constant(0.0,shape=[SEQ_LENGTH]))
            #self.eval_target = tf.zeros((SEQ_LENGTH,1)) 
            self.target = tf.placeholder(tf.float32,[None,config.MaxNumSteps])
            loss = tf.Variable(0.,trainable=False)
            x_split = tf.split(0,config.batch_size,self.input_data)
            y_split = tf.split(0,config.batch_size,self.target)
            x_eval_split = tf.split(0,1,self.eval_input_data)# split the evaluation input by the number of time steps
            #x_eval_split = [self.eval_input_data for _ in range(SEQ_LENGTH)]

            w = tf.Variable(tf.random_normal([config.hidden_size,config.FC_Units],stddev=0.1),trainable=True)
            b = tf.Variable(tf.constant(0.0,shape=[config.FC_Units]),trainable=True)
            #w_2 = tf.Variable(tf.random_normal([config.FC_Units,config.MaxNumSteps],stddev=0.1),trainable=True)
            #b_2 = tf.Variable(tf.constant(0.0,shape=[config.MaxNumSteps]),trainable=True)
            w_2 = tf.Variable(tf.random_normal([config.FC_Units,1],stddev=0.1),trainable=True)
            b_2 = tf.Variable(tf.constant(0.0,shape=[1]),trainable=True)
            #Initialize basic lstm cell
            lstm = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,state_is_tuple=True)
            #ops, states = tf.nn.rnn(lstm,x_split,dtype=tf.float32)
            #lstm_multi = tf.nn.rnn_cell.MultiRNNCell([lstm]*config.num_layers,state_is_tuple=True)
            #ops, states = tf.nn.rnn(lstm_multi,x_split,dtype=tf.float32)
            lstm_init = lstm.zero_state(config.batch_size,tf.float32)
            for ii in range(config.MaxNumSteps):
                if ii == 0:
                    output, output_state = lstm(tf.reshape(self.input_data[:,ii],\
                    shape=(config.batch_size,1)), lstm_init)
                else:
                    train_scope.reuse_variables()
                    output, output_state = lstm(tf.reshape(self.input_data[:,ii],\
                    shape=(config.batch_size,1)), output_state)

                transform1 = tf.nn.elu(tf.matmul(output,w)+b)
                dropout = tf.nn.dropout(transform1,keep_prob=0.75)
                transform2 = tf.nn.elu(tf.matmul(dropout,w_2)+b_2)
                loss += tf.reduce_mean((transform2-self.target[:,ii])**2)
            loss += 0.1* tf.nn.l2_loss(w) + 0.1*tf.nn.l2_loss(w_2) + 0.1*tf.nn.l2_loss(b) + 0.1*tf.nn.l2_loss(b_2)
            '''
            #Compute loss
            for op,target in zip(ops,y_split):
                transform = tf.nn.elu(tf.matmul(op,w)+b)
                drop_out = tf.nn.dropout(transform,keep_prob=0.6)
                fc_layer2 = tf.nn.elu(tf.matmul(drop_out,w_2)+b_2)
                self.output.append(fc_layer2)
                loss += self.loss_function(transform,target,w,b)
            '''
            #Use the variables above to also unravel the eval node
            self.loss = loss
            self.lr = tf.Variable(0.0, trainable=False)
            '''
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          config.max_grad_norm)
            #optimizer = tf.train.GradientDescentOptimizer(self.lr)
            #optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.5)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            '''
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
            #Eval network
            train_scope.reuse_variables()
        #with tf.variable_scope(scope,reuse=True):
        #with tf.variable_scope("rnnlhc_eval") as scope:
            eval_target_lst = []
            lstm_init = lstm.zero_state(1,tf.float32) #Init for batch size 1
            #scope.reuse_variables()
            for tstep in range(config.MaxNumSteps):
                if tstep == 0:
                    output, output_state = lstm(tf.reshape(self.eval_input_data[0,tstep],shape=(1,1)),lstm_init)
                    eval_target_lst.append(tf.reshape(self.eval_input_data[0,tstep],shape=(1,1)))
                    #scope.reuse_variables()
                else:
                    output, output_state = lstm(tf.reshape(self.eval_input_data[0,tstep],shape=(1,1)),output_state)
                    transform1 = tf.nn.elu(tf.matmul(output,w)+b)
                    transform2 = tf.nn.elu(tf.matmul(transform1,w_2)+b_2)
                    eval_target_lst.append(transform2)
            self.eval_target = tf.pack(eval_target_lst)


    def loss_function(self,ip,op,w,b):
        return tf.reduce_mean((ip-op)**2) + tf.nn.l2_loss(w) + tf.nn.l2_loss(b)


    def generate_data(self,num=None,data_type=1):
            x_data_list = np.zeros((num,self.config.MaxNumSteps+1))
            y_data_list = np.zeros((num,self.config.MaxNumSteps+1))
            for j in range(0,num):
                    #generate randome sequence equal to number of points
                    x_data = np.array([np.linspace(0,4*np.pi,num = self.config.MaxNumSteps+1)] )
                    if data_type == 1: #Sinusoid
                        y_data = self.sinusoid(x_data)
                    elif data_type == 2: #Sinusoid with large noise
                        y_data = self.sinusoid2(x_data)
                    x_data_list[j,:] = x_data
                    y_data_list[j,:] = y_data
            
            return x_data_list,y_data_list
    
    def sinusoid2(self,x):
        return 0.5*np.sin(x*np.random.rand()/10.) + np.random.rand()/100.
        #return 0.5*np.sin(x) + 0.25

    def sinusoid(self,x):
        return 0.5*np.sin(x) + np.random.rand()/10.

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def plot_data(self,data):
        fig = plt.figure()
        for ii in range(data.shape[0]):
            plt.plot(data[ii,:])
            plt.hold(True)
        return fig

    def run_model(self,sess,m,data,eval_op,verbose=True):
      cost,_ = sess.run([m.loss,eval_op],{m.input_data: data[:,:-1],m.target: data[:,1:]})
      return cost

    def eval_model(self,sess,m,data,eval_op):
      output = sess.run([eval_op],{m.eval_input_data:data[:,:-1]})
      return output


class TestConfig(object):
  """Tiny config, for testing."""
  learning_rate = 1e-4
  max_grad_norm = 0.1
  num_layers = 2
  MaxNumSteps = 12 
  feat_dims = 1
  hidden_size = 20
  max_epoch = 1
  lr_decay = 0.
  batch_size = 20
  num_layers = 2
  FC_Units = 60

