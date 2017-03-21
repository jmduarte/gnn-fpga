import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython

#SEQ_LENGTH = 10

class testrnn:
    def __init__(self,config):
        self.config = config
        self.dim = 2
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.variable_scope("rnnlhc") as train_scope:
            self.input_data = tf.placeholder(tf.float32,[config.batch_size,config.MaxNumSteps-1,self.dim])
            self.eval_input_data = tf.placeholder(tf.float32,[config.batch_size,config.MaxNumSteps-1,self.dim])
            self.target = tf.placeholder(tf.float32,[config.batch_size,config.MaxNumSteps-1,self.dim])
            loss = tf.Variable(0.,trainable=False)

            w = tf.Variable(tf.random_normal([config.hidden_size,config.FC_Units],stddev=0.1),trainable=True)
            b = tf.Variable(tf.constant(0.0,shape=[config.FC_Units]),trainable=True)
            w_2 = tf.Variable(tf.random_normal([config.FC_Units,self.dim],stddev=0.1),trainable=True)
            b_2 = tf.Variable(tf.constant(0.0,shape=[self.dim]),trainable=True)
            #Initialize basic lstm cell
            #lstm = [tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,state_is_tuple=True) for d in range(3)]
            #lstm = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size,state_is_tuple=True)
            lstm = tf.nn.rnn_cell.GRUCell(config.hidden_size)
            #output_state = [l.zero_state(config.batch_size,tf.float32) for l in lstm]
            #output_state = [None]*3
            train_output = []
            #output = [tf.zeros((config.batch_size,config.hidden_size),tf.float32)]*3
            train_interim_output = tf.Variable(tf.constant(0.,shape=[config.batch_size,self.dim]),trainable=False,name='TrainInterim')
            x_transpose = tf.transpose(self.input_data,[1,0,2])
            x_reshape = tf.reshape(x_transpose,[-1,self.dim])
            x_split = tf.split(0,config.MaxNumSteps-1,x_reshape)
            output, states = tf.nn.rnn(lstm,x_split,dtype=tf.float32)
            '''
            for ii in range(config.MaxNumSteps):
                print ("The value of ii is {}".format(ii))
                for jj,ll in enumerate(lstm):
                    print jj
                    if ii ==0 and jj ==1:
                        train_scope.reuse_variables()
                    output[jj], output_state[jj] = ll(tf.gather(tf.reshape(self.input_data[:,ii,jj],shape=(config.batch_size,1))), output_state[jj])
                concat_output = tf.concat(concat_dim=1,values=output)
                '''
            for ii,output_step in enumerate(output):
                transform1 = tf.nn.elu(tf.matmul(output_step,w)+b)
                transform2 = tf.nn.elu(tf.matmul(transform1,w_2)+b_2)
                #squared_op = (transform2[:,1:] - self.target[:,ii,1:])**2
                squared_op = (transform2 - self.target[:,ii,:])**2
                loss += tf.reduce_mean(squared_op)
                train_interim_output = tf.transpose(squared_op)
                train_output.append(transform2)
            self.train_output = tf.pack(train_output)
            #Use the variables above to also unravel the eval node
            loss = loss +  0.2*tf.reduce_sum(tf.abs(w)) + 0.2*tf.reduce_sum(tf.abs(w_2))
            self.loss = loss
            #self.loss = tf.reduce_mean(loss)
            #self.lr = tf.Variable(0.0, trainable=False,name='LR')
            self.lr = tf.Variable(1e-3, trainable=False,name='LR')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
            #self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss)
            self.train_interim_output = train_interim_output
            #Eval network
            train_scope.reuse_variables()
            eval_target_lst = []
            #output_state = [l.zero_state(1,tf.float32) for l in lstm]
            #output = [tf.zeros((1,config.hidden_size),tf.float32)]*3
            #scope.reuse_variables()
            '''
            for tstep in range(config.MaxNumSteps):
                print tstep
                for jj,ll in enumerate(lstm):
                    print jj
                    output[jj], output_state[jj] = ll(tf.reshape(self.eval_input_data[0,tstep,jj],shape=(1,1)),output_state[jj])
                concat_output = tf.concat(concat_dim=1,values=output)
            '''
            eval_transpose = tf.transpose(self.eval_input_data,[1,0,2])
            eval_reshape = tf.reshape(eval_transpose,[-1,self.dim])
            eval_split = tf.split(0,config.MaxNumSteps-1,eval_reshape)
            output, states = tf.nn.rnn(lstm,eval_split,dtype=tf.float32)
            eval_target_lst = []
            for tstep,output_step in enumerate(output):
                #eval_target_lst.append(tf.reshape(self.eval_input_data[0,tstep,:],shape=(1,self.dim)))
                transform1 = tf.nn.elu(tf.matmul(output_step,w)+b)
                transform2 = tf.nn.elu(tf.matmul(transform1,w_2)+b_2)
                eval_target_lst.append(transform2)
            euclidean_loss = np.zeros([self.dim,config.batch_size])
            for tstep in range(config.MaxNumSteps-1):
                euclidean_loss = euclidean_loss + (tf.transpose(eval_target_lst[tstep]) - tf.reshape(self.eval_input_data[:,tstep,:],shape=(self.dim,config.batch_size)))**2

            euclidean_loss /= config.MaxNumSteps-1
            self.eucl_loss = tf.reduce_mean(euclidean_loss,1)
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
      cost,_, train_output,train_interim_output,summ = sess.run\
      ([m.loss,eval_op,m.train_output,m.train_interim_output,m.summary],\
      {m.input_data: data[:,:-1,:],m.target: data[:,1:,:]})
      return cost,summ

    def eval_model(self,sess,m,data,eval_op,eucl_l):
      #data = data.reshape(1,self.config.MaxNumSteps+1)
      output,eucl_l = sess.run([eval_op,eucl_l],{m.eval_input_data:data[:,:-1,:]})
      return output, eucl_l


class TestConfig(object):
  """Tiny config, for testing."""
  learning_rate = 1e-4
  max_grad_norm = 0.1
  num_layers = 2
  MaxNumSteps = 12
  feat_dims = 1
  hidden_size =20
  max_epoch = 1
  lr_decay = 0.
  batch_size =  100
  num_layers = 2
  FC_Units = 20
  lam = 0.0

