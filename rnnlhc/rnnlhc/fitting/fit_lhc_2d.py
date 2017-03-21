import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from rnnlhc.fitting.lstm_lhc_2d import testrnn
from rnnlhc.fitting.lstm_lhc_2d import TestConfig
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
   parser.add_argument("--num_steps",default=12,type=int,help="Num Steps")
   args = parser.parse_args()
   np.random.seed(1234)
   data = np.load(args.npy_data)
   BD= BatchNpyData2(data)
   import IPython; IPython.embed()
   config = TestConfig()
   config.MaxNumSteps = args.num_steps
   m = testrnn(config)
   test, filtered_test_data, rand_int, test_idx  = BD.sample_batch(m.config.MaxNumSteps,None,m.config.batch_size)
   test[:,:,0] = test[:,:,0]/test[:,:,2] #Dividing Rho Phi by Rho
   filtered_test_data[:,:,0] = filtered_test_data[:,:,0]/filtered_test_data[:,:,2]
   test, max_data  = pre_process(test)
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
            data, filtered_data, rand_int, _ = BD.sample_batch(m.config.MaxNumSteps,test_idx,m.config.batch_size)
            data[:,:,0] = data[:,:,0]/data[:,:,2] #Dividing Rho Phi by Rho
            data, _  = pre_process(data,max_data)
            cost,summ = m.run_model(sess,m,data[:,:,:-1],m.train_op)
            summary_writer.add_summary(summ,ii)
            cost_lst.append(cost)
            if np.mod(ii,100) == 0:
                print("cost is {}".format(cost))
                eval_list =[]
                output_list = []
                output,eucl_l = m.eval_model(sess,m,test[:,:,:-1],m.eval_target,m.eucl_loss)
                print("Euclidean loss is {}".format(eucl_l))
                output_list.append(output)
        output = output.transpose(1,0,2)
        #Adding the third dimension
        output = np.c_[output,test[:,1:,-1].reshape(m.config.batch_size,m.config.MaxNumSteps-1,-1)]
        plt.plot(cost_lst)
        plt.title('Cost vs iterations')
        plt.savefig('RNN_train_1.png')
        print("Saving output dump")
        np.savez('../data/rnndump.npy',test=test,output=output,filt_test=filtered_test_data)
        print("Making plots")
        for ii in np.arange(20):
          fig = plt.figure()
          ax = fig.gca(projection='3d')
          ax.plot(test[ii,1:,0],test[ii,1:,1],test[ii,1:,2], 'r*',linewidth=3)
          ax.hold(True)
          ax.plot(test[ii,1:,0],test[ii,1:,1],test[ii,1:,2],'r' )
          ax.plot(output[ii,:,0],output[ii,:,1],output[ii,:,2],'g+',linewidth=3)
          ax.plot(output[ii,:,0],output[ii,:,1],output[ii,:,2],'g')
          ax.hold(True)
          ax.view_init(elev=18, azim=-27)
          ax.dist=9
          plt.savefig('png/3dacts_'+str(ii)+'.png')
          proj_2d_plot(test,output,savestr='proj_lstm_')
          proj_2d_plot(test,filtered_test_data,savestr='proj_kf_')
