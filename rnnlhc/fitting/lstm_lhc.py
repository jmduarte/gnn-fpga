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
            self.input_data = tf.placeholder(tf.float32,[config.batch_size,config.MaxNumSteps-1,3])
            self.eval_input_data = tf.placeholder(tf.float32,[1,config.MaxNumSteps-1,3])
            self.target = tf.placeholder(tf.float32,[config.batch_size,config.MaxNumSteps-1,3])
            loss = tf.Variable(0.,trainable=False)

            w = tf.Variable(tf.random_normal([config.hidden_size,config.FC_Units],stddev=0.1),trainable=True)
            b = tf.Variable(tf.constant(0.0,shape=[config.FC_Units]),trainable=True)
            w_2 = tf.Variable(tf.random_normal([config.FC_Units,3],stddev=0.1),trainable=True)
            b_2 = tf.Variable(tf.constant(0.0,shape=[3]),trainable=True)
            #Initialize basic lstm cell
            lstm = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,state_is_tuple=True)
            lstm_init = lstm.zero_state(config.batch_size,tf.float32)
            train_output = []
            train_interim_output = tf.Variable(tf.constant(0.,shape=[config.batch_size,3]),trainable=False,name='TrainInterim')
            for ii in range(config.MaxNumSteps):
                if ii == 0:
                    output, output_state = lstm(tf.reshape(self.input_data[:,ii,:],\
                    shape=(config.batch_size,3)), lstm_init)
                else:
                    train_scope.reuse_variables()
                    output, output_state = lstm(tf.reshape(self.input_data[:,ii,:],\
                    shape=(config.batch_size,3)), output_state)

                transform1 = tf.nn.elu(tf.matmul(output,w)+b)
                transform2 = tf.nn.elu(tf.matmul(transform1,w_2)+b_2)
                squared_op = (tf.transpose(transform2) - tf.reshape(self.target[:,ii,:],shape=(3,config.batch_size)))**2
                loss += tf.reduce_mean(squared_op)
                train_interim_output = tf.transpose(squared_op) 
                train_output.append(transform2)
            self.train_output = tf.pack(train_output)
            #Use the variables above to also unravel the eval node
            self.loss = loss
            #self.loss = tf.reduce_mean(loss)
            self.lr = tf.Variable(0.0, trainable=False,name='LR')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
            self.train_interim_output = train_interim_output
            #Eval network
            train_scope.reuse_variables()
            eval_target_lst = []
            lstm_init = lstm.zero_state(1,tf.float32) #Init for batch size 1
            #scope.reuse_variables()
            for tstep in range(config.MaxNumSteps):
                if tstep == 0:
                    output, output_state = lstm(tf.reshape(self.eval_input_data[0,tstep,:],shape=(1,3)),lstm_init)
                    eval_target_lst.append(tf.reshape(self.eval_input_data[0,tstep,:],shape=(1,3)))
                    #scope.reuse_variables()
                else:
                    output, output_state = lstm(tf.reshape(self.eval_input_data[0,tstep,:],shape=(1,3)),output_state)
                    transform1 = tf.nn.elu(tf.matmul(output,w)+b)
                    transform2 = tf.nn.elu(tf.matmul(transform1,w_2)+b_2)
                    eval_target_lst.append(transform2)
            euclidean_loss = 0.

            for tstep in range(config.MaxNumSteps-1):
                euclidean_loss = euclidean_loss + (tf.transpose(eval_target_lst[tstep]) - \
                tf.reshape(self.eval_input_data[0,tstep,:],shape=(1,3)))**2

            self.eucl_loss = euclidean_loss
            self.eval_target = tf.pack(eval_target_lst)
            w1_summary = tf.histogram_summary('w1',w)
            w2_summary = tf.histogram_summary('w2',w_2)
            self.summary = tf.merge_all_summaries()


    def init_logging(self,sess):
        w_summary_t = tf.image_summary('w1',w)
        self.w_summary_t = w_summary_t
        return summary_op

    def save_summary(self,sess,smry,step):
        summaryWriter = tf.train.SummaryWriter('/home/mudigonda/Projects/rnnlhc/rnnlhc/fitting/Logs',sess.graph)
        #assumes smry is a list
        if not type(smry) is list:
            smry = [smry]
        for sm in smry:
            summaryWriter.add_summary(sm,step)


    def loss_function(self,ip,op,w,b):
        return tf.reduce_mean((ip-op)**2) + tf.nn.l2_loss(w) + tf.nn.l2_loss(b)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def plot_data(self,data):
        fig = plt.figure()
        for ii in range(data.shape[0]):
            plt.plot(data[ii,:])
            plt.hold(True)
        return fig

    def run_model(self,sess,m,data,eval_op,verbose=True):
      import IPython; IPython.embed()
      cost,_, train_output,train_interim_output,summ = sess.run\
      ([m.loss,eval_op,m.train_output,m.train_interim_output,m.summary],\
      {m.input_data: data[:,:-1,:],m.target: data[:,1:,:]})
      return cost,summ

    def eval_model(self,sess,m,data,eval_op,eucl_l):
      data = data.reshape(1,self.config.MaxNumSteps+1)
      output,eucl_l = sess.run([eval_op,eucl_l],{m.eval_input_data:data[:,:-1,:]})
      return output, eucl_l


class TestConfig(object):
  """Tiny config, for testing."""
  learning_rate = 1e-4
  max_grad_norm = 0.1
  num_layers = 2
  MaxNumSteps = 12
  feat_dims = 1
  hidden_size =2
  max_epoch = 1
  lr_decay = 0.
  batch_size =  200
  num_layers = 2
  FC_Units = 10
  lam = 0.0

