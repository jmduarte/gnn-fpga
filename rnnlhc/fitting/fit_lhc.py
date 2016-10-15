import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rnnlhc.fitting.lstm_lhc import testrnn
from rnnlhc.fitting.lstm_lhc import TestConfig
import tensorflow as tf
import pickle
import argparse
from rnnlhc.fitting import BatchData
import json

def get_train_batch(data,n_samples):
    indices = np.random.randint(0,np.size(data,0),n_samples)
    return data[indices,...]

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Arguments for fitting")
   parser.add_argument("--niter",default=100,type=int,help="Number of iterations")
   parser.add_argument("--json_data",default='../data/EventDump_10Ktracks.json',type=str,help="Json data path")
   args = parser.parse_args()
   np.random.seed(1234)
   json_data = open(args.json_data,'r').read()
   parsed_json_data = json.loads(json_data)
   BD = BatchData.BatchData(parsed_json_data)
   config = TestConfig()
   m = testrnn(config)
   cost_lst = []
   with tf.Session() as sess:
        tf.set_random_seed(1234)
        summary_writer = tf.train.SummaryWriter('Logs/')
        sess.run(tf.initialize_all_variables())
        for ii in range(args.niter):
            m.assign_lr(sess,config.learning_rate)
            #ind,data = m.generate_data(m.config.batch_size,2)
            data,rand_int = BD.sample_batch(m.config.MaxNumSteps,m.config.batch_size)
            cost,summ = m.run_model(sess,m,data,m.train_op)
            summary_writer.add_summary(summ,ii)
            cost_lst.append(cost)
            if np.mod(ii,100) == 0:
                print("cost is {}".format(cost))
                eval_list =[]
                output_list = []
                for kk in range(10): #Let's evaluate on 5 inputs
                    #ind,eval_data = m.generate_data(1,2) #Eval data
                    output,eucl_l = m.eval_model(sess,m,test[kk,:],m.eval_target,m.eucl_loss)
                    print("Euclidean loss is {}".format(eucl_l))
                    output_list.append(output)
        plt.plot(cost_lst)
        plt.title('Cost vs iterations')
        plt.savefig('RNN_train_1.png')
        plt.clf()
        for kk in range(10):
            plt.plot(range(config.MaxNumSteps+1),test[kk,:],'r',label='Data')
            plt.hold(True)
            plt.plot(range(config.MaxNumSteps+1),test[kk,:],'g+')
            plt.plot(range(1,config.MaxNumSteps+1),np.array(output_list[kk]).flatten().T,'b',label='Reconstr')
            plt.plot(range(1,config.MaxNumSteps+1),np.array(output_list[kk]).flatten().T,'k*')
            plt.legend()
            axes = plt.gca()
            axes.set_ylim([-0.6,0.6])
            #axes.set_yscale('log')
            plt.title('reconstruction of trajectories')
            plt.savefig('reconstr' + str(kk) + '.png')
            plt.clf()
        plt.plot(data.T)
        plt.title('Example Data batch')
        plt.savefig('data.png')

