import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rnnlhc.fitting.lstm_lhc import testrnn
from rnnlhc.fitting.lstm_lhc import TestConfig
import tensorflow as tf
import pickle
import argparse
from rnnlhc.fitting.BatchData import BatchNpyData2
from utilities import proj_2d_plot,pre_process


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Arguments for fitting")
   parser.add_argument("--niter",default=100,type=int,help="Number of iterations")
   parser.add_argument("--json_data",default='../data/EventDump_10Ktracks.json',type=str,help="Json data path")
   parser.add_argument("--npy_data",default='../data/ET_muons_10K_0000.npy',type=str,help="NPY data")
   args = parser.parse_args()
   np.random.seed(1234)
   data = np.load(args.npy_data)
   BD= BatchNpyData2(data)
   config = TestConfig()
   m = testrnn(config)
   data, filtered_data, rand_int = BD.sample_batch(m.config.MaxNumSteps,1000)
   data, max_data = pre_process(data)
   filtered_data, _ = pre_process(filtered_data,max_data)
   test, filtered_test_data, rand_int = BD.sample_batch(m.config.MaxNumSteps,m.config.batch_size)
   test, _ = pre_process(test,max_data)
   filtered_test_data, _ = pre_process(filtered_test_data,max_data)
   cost_lst = []
   tf_config = tf.ConfigProto()
   tf_config.gpu_options.allow_growth = True
   with tf.Session(config=tf_config) as sess:
        tf.set_random_seed(1234)
        summary_writer = tf.train.SummaryWriter('Logs/')
        sess.run(tf.initialize_all_variables())
        for ii in range(args.niter):
            m.assign_lr(sess,config.learning_rate)
            #ind,data = m.generate_data(m.config.batch_size,2)
            data, filtered_data, rand_int = BD.sample_batch(m.config.MaxNumSteps,m.config.batch_size)
            data, _  = pre_process(data,max_data)
            cost,summ = m.run_model(sess,m,data,m.train_op)
            summary_writer.add_summary(summ,ii)
            cost_lst.append(cost)
            if np.mod(ii,100) == 0:
                print("cost is {}".format(cost))
                eval_list =[]
                output_list = []
                output,eucl_l = m.eval_model(sess,m,test,m.eval_target,m.eucl_loss)
                print("Euclidean loss is {}".format(eucl_l))
                output_list.append(output)
        plt.plot(cost_lst)
        plt.title('Cost vs iterations')
        plt.savefig('RNN_train_1.png')
        np.savez('../data/rnndump.npy',test=test,output=output,filt_test=filtered_test_data)
        for ii in np.arange(20):
          fig = plt.figure()
          ax = fig.gca(projection='3d')
          ax.plot(test[ii,1:,0],test[ii,1:,1],test[ii,1:,2], 'r*',linewidth=3)
          ax.hold(True)
          ax.plot(test[ii,1:,0],test[ii,1:,1],test[ii,1:,2],'r' )
          ax.plot(output[:,ii,0],output[:,ii,1],output[:,ii,2],'g+',linewidth=3)
          ax.plot(output[:,ii,0],output[:,ii,1],output[:,ii,2],'g')
          ax.hold(True)
          ax.view_init(elev=18, azim=-27)
          ax.dist=9
          plt.savefig('png/3dacts_'+str(ii)+'.png')
          proj_2d_plot(test,output,savestr='proj_lstm_')
          proj_2d_plot(test,np.transpose(filtered_test_data,(1,0,2)),savestr='proj_kf_')
