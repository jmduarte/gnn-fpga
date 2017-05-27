from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from acts_data import load_data, clean_data
from lstm_lhc_2d import testrnn, TestConfig
from utilities import proj_2d_plot

input_file = '/Users/mudigonda/Data/KFTest_all.npy'
# Load the raw data
raw_data = load_data(input_file)
num_raw_tracks = len(raw_data.true_theta)
# Clean the data
data = clean_data(raw_data, fix_phi=True)
num_tracks = len(data.true_theta)
print('Number of raw tracks: %i, cleaned tracks: %i' % (num_raw_tracks, num_tracks))

TC = TestConfig()
Max_Train_Sample_Idx = 24362
TC.MaxNumSteps = 9
LSTM_LHC = testrnn(TC)


#Normalization
norm_phi = np.linalg.norm(data.phi)
norm_z = np.linalg.norm(data.z)
norm_r = np.linalg.norm(data.r)
data.phi = data.phi/norm_phi
data.z = data.z/norm_z
data.r = data.r/norm_r
data.KF_z = data.KF_z/norm_z
data.KF_r = data.KF_r/norm_r
data.KF_phi = data.KF_phi/norm_phi

cost_lst = []
train_data = np.dstack((data.phi[:Max_Train_Sample_Idx],data.z[:Max_Train_Sample_Idx]))
test_data = np.dstack((data.phi[Max_Train_Sample_Idx:],data.z[Max_Train_Sample_Idx:]))
kf_data = np.dstack((data.KF_phi[Max_Train_Sample_Idx:],data.KF_z[Max_Train_Sample_Idx:]))

with tf.Session() as sess:
        tf.set_random_seed(1234)
        summary_writer = tf.train.SummaryWriter('Logs/')
        sess.run(tf.initialize_all_variables())
        LSTM_LHC.assign_lr(sess,TC.learning_rate)
        for ii in range(7000):                    
            batch_data_idx = np.random.randint(0,Max_Train_Sample_Idx,TC.batch_size)
            #batch_data = np.dstack((data.r[idx],data.phi[idx],data.z[idx]))
            batch_data = np.dstack((data.phi[batch_data_idx],data.z[batch_data_idx]))
            cost,summ = LSTM_LHC.run_model(sess,LSTM_LHC,batch_data,LSTM_LHC.train_op)
            summary_writer.add_summary(summ,ii)
            cost_lst.append(cost)
            if np.mod(ii,100) == 0:
                print("cost is {}".format(cost))
                eval_list =[]
                output_list = []
                output,eucl_l = LSTM_LHC.eval_model(sess,LSTM_LHC,test_data,LSTM_LHC.eval_target,LSTM_LHC.eucl_loss)
                print("Euclidean loss is {}".format(eucl_l))
                output_list.append(output)
       
test_data_appended = np.dstack((data.r[Max_Train_Sample_Idx:],test_data))
output_data_appended = np.dstack((data.r[Max_Train_Sample_Idx:][:,1:,np.newaxis],output.transpose(1,0,2)))
kf_data = np.dstack((data.KF_r[Max_Train_Sample_Idx:],kf_data))

plt.figure()
plt.plot(cost_lst)
plt.title('Cost vs iterations')
plt.savefig('RNN_train_1.png')
print("Saving output dump")
#np.savez('../data/rnndump.npy',test=test_data_appended,output=output,filt_test=filtered_test_data)
print("Making plots")
for ii in np.arange(20):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(test_data_appended[ii,1:,0],test_data_appended[ii,1:,1],test_data_appended[ii,1:,2], 'r*',linewidth=3)
    ax.plot(test_data_appended[ii,1:,0],test_data_appended[ii,1:,1],test_data_appended[ii,1:,2],'r' )
    ax.plot(output_data_appended[ii,:,0],output_data_appended[ii,:,1],output_data_appended[ii,:,2],'g+',linewidth=3)
    ax.plot(output_data_appended[ii,:,0],output_data_appended[ii,:,1],output_data_appended[ii,:,2],'g')
    #ax.axis([output_data_appended.min(),output_data_appended.max(),output_data_appended.min(),output_data_appended.max()])
    ax.view_init(elev=18, azim=-27)
    ax.dist=9
    ax.set_xlabel('Rho')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Z')
    plt.savefig('png/3dacts_'+str(ii)+'.png')
    plt.close()
proj_2d_plot(test_data_appended,output_data_appended,savestr='proj_lstm_')
proj_2d_plot(test_data_appended,kf_data,savestr='proj_kf_')
print("KF Residual {}".format(np.linalg.norm(np.linalg.norm(test_data_appended[:,1:,:] - kf_data[:,1:,:], axis=1),axis=1).mean()))
print("LSMT Residual {} ".format(np.linalg.norm(np.linalg.norm(test_data_appended[:,1:,:] - output_data_appended, axis=1),axis=1).mean()))
print("LSTM Residual Std {}".format(np.linalg.norm(np.linalg.norm(test_data_appended[:,1:,:] - output_data_appended, axis=1),axis=1).std()))
print("KF Residual Std {}".format(np.linalg.norm(np.linalg.norm(test_data_appended[:,1:,:] - kf_data[:,1:,:], axis=1),axis=1).std()))
