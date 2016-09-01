import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rnnlhc.fitting.simpler_lstm import testrnn
from rnnlhc.fitting.simpler_lstm import TestConfig
import tensorflow as tf

if __name__ == "__main__":
   config = TestConfig()
   m = testrnn(config)
   cost_lst = []
   with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for ii in range(2000):
            m.assign_lr(sess,config.learning_rate)
            ind,data = m.generate_data(m.config.batch_size,2)
            cost = m.run_model(sess,m,data,m.train_op)
            cost_lst.append(cost)
            if np.mod(ii,100) == 0:
                print("cost is {}".format(cost))
                eval_list =[]
                output_list = []
                np.random.seed(12345)
                for kk in range(5): #Let's evaluate on 5 inputs
                    ind,eval_data = m.generate_data(1,2) #Eval data
                    output,eucl_l = m.eval_model(sess,m,eval_data,m.eval_target,m.eucl_loss)
                    print("Euclidean loss is {}".format(eucl_l))
                    eval_list.append(eval_data)
                    output_list.append(output)
        plt.plot(cost_lst)
        plt.title('Cost vs iterations')
        plt.savefig('RNN_train_1.png')
        plt.clf()
        for kk in range(5):
            plt.plot(eval_list[kk].T,'r',label='Data')
            plt.hold(True)
            plt.plot(eval_list[kk].T,'g+')
            plt.plot(np.array(output_list[kk]).flatten().T,'b',label='Reconstr')
            plt.plot(np.array(output_list[kk]).flatten().T,'k*')
            plt.legend()
            plt.title('reconstruction of trajectories')
            plt.savefig('reconstr' + str(kk) + '.png')
            plt.clf()
        plt.plot(data.T)
        plt.title('Example Data batch')
        plt.savefig('data.png')

