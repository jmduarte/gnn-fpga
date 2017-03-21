'''
Script that takes the output of 2000 test samples and plots
@Mayur Mudigonda, Feb 13 2017
'''
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if __name__ == "__main__":
   AP = argparse.ArgumentParser()
   AP.add_argument("--fname",default=None,type=str,help="Name of output npz file")
   AP.add_argument("--n_hits",default=31,type=int,help="Number of hits")
   args = AP.parse_args()
   data = np.load(args.fname)
   test = data['test']
   output = data['output']
   #output = output.transpose(1,0,2)
   filtered = data['filt_test']
   #PLotting RPhi
   plt.clf()
   nbins = 100
   rphi_bins = np.histogram(np.concatenate([test[:,1:,0].flatten()-filtered[:,1:,0].flatten(),test[:,1:,0].flatten()-output[:,:,0].flatten()]),bins=nbins)
   hist_rphi_kf = plt.hist(test[:,1:,0].flatten() - filtered[:,1:,0].flatten(),bins=rphi_bins[1],label='Meas RPhi - KF RPhi',color='green',alpha=0.3)
   print hist_rphi_kf
   plt.hold(True)
   #hist_rphi_lstm = plt.hist(test[:,1:,0].flatten() - output[:,:,0].flatten(),bins=RPhi_bins,label='Meas RPhi - LSTM RPhi',color='red')
   hist_rphi_lstm = plt.hist(test[:,1:,0].flatten() - output[:,:,0].flatten(),bins=rphi_bins[1],label='Meas RPhi - LSTM RPhi',color='red',alpha=0.7)
   print hist_rphi_lstm
   plt.legend(loc=0)
   plt.xlabel('Meas RPhi - Predicted RPhi')
   plt.savefig('Hist_diff_RPhi.png')
   #PLotting Z
   plt.clf()
   z_bins = np.histogram(np.concatenate([test[:,1:,1].flatten()-filtered[:,1:,1].flatten(),test[:,1:,1].flatten()-output[:,:,1].flatten()]),bins=nbins)
   hist_z_kf = plt.hist(test[:,1:,1].flatten() - filtered[:,1:,1].flatten(),bins=z_bins[1],label='Meas Z - KF Z',color='green',alpha=0.3)
   print hist_z_kf
   plt.hold(True)
   hist_z_lstm = plt.hist(test[:,1:,1].flatten() - output[:,:,1].flatten(),bins=z_bins[1],label='Meas Z - LSTM Z',color='red',alpha=0.7)
   print hist_z_lstm
   plt.legend(loc=0)
   plt.xlabel('Meas Z - Predicted Z')
   plt.savefig('Hist_diff_Z.png')
   #Plotting Z LSTM - KF
   plt.clf()
   Diff_Z=plt.hist(filtered[:,1:,1].flatten() - output[:,:,1].flatten(),label='KF Z - LSTM Z',color='blue')
   print Diff_Z
   plt.legend(loc=0)
   plt.xlabel('KF Z - LSTM Z')
   plt.savefig('Hist_diff_KF_LSTM_Z.png')
   #Plotting RPhi LSTM - KF
   plt.clf()
   Diff_RPhi = plt.hist(filtered[:,1:,0].flatten() - output[:,:,0].flatten(),label='KF RPhi - LSTM RPhi',color='blue')
   print Diff_RPhi
   plt.legend(loc=0)
   plt.xlabel('KF RPhi - LSTM RPhi')
   plt.savefig('Hist_diff_KF_LSTM_RPhi.png')

   #Printing the eucl distance
   print("The Eucl distance beween Meas and LSTM {}".format(np.linalg.norm(test[:,1:,:]-output)))
   print("The Eucl distance beween Meas and KF {}".format(np.linalg.norm(test[:,1:,:]-filtered[:,1:,:])))
   print("The Eucl distance beween KF and LSTM {}".format(np.linalg.norm(output-filtered[:,1:,:])))
   kf_diff = ((test[:,1:,:] - filtered[:,1:,:])**2).sum(axis=2).sum(axis=1)
   lstm_diff = ((test[:,1:,:] - output)**2).sum(axis=2).sum(axis=1)
   kf_lstm_diff = kf_diff - lstm_diff

   #Plotting error distance of KF vs R
   plt.clf()
   kf_diff_per_layer = ((test[:,1:,:] - filtered[:,1:,:])**2).sum(axis=2)
   lstm_diff_per_layer = ((test[:,1:,:] - output)**2).sum(axis=2)
   plt.plot(kf_diff_per_layer.flatten(),test[:,1:,-1].flatten(),'r*',label='KF Eucl loss versus Rho')
   plt.legend(loc='upper right')
   plt.xlabel('Meas - KF')
   plt.ylabel('Rho')
   plt.savefig('Meas_KF_vs_R.png')
   #plt.hold(True)
   #plt.plot(kf_diff_per_layer.flatten(),test[:,1:,-1].flatten(),'r')
   #Plotting error distance of LSTM vs R
   plt.clf()
   plt.plot(lstm_diff_per_layer.flatten(),test[:,1:,-1].flatten(),'g*',label='LSTM Eucl loss versus Rho')
   #plt.plot(lstm_diff_per_layer.flatten(),test[:,1:,-1].flatten(),'g')
   plt.legend(loc='upper right')
   plt.xlabel('Meas - LSTM')
   plt.ylabel('Rho')
   plt.savefig('Meas_LSTM_vs_R.png')
   #Plotting error distance of KF vs detector layer
   plt.clf()
   detector_layer = np.tile(np.arange(args.n_hits-1),(test.shape[0],1))
   plt.plot(detector_layer.flatten(),kf_diff_per_layer.flatten(),'r*',label='KF Eucl loss vs layer number')
   plt.legend(loc='upper right')
   plt.xlabel('Layer number')
   plt.ylabel('Meas - KF Eucl loss')
   plt.savefig('Meas_KF_vs_Layer.png')
   #Plotting error distance of LSTM vs detector layer
   plt.clf()
   plt.plot(detector_layer.flatten(),lstm_diff_per_layer.flatten(),'g*',label='KF Eucl loss vs layer number')
   plt.legend(loc='upper right')
   plt.xlabel('Layer number')
   plt.ylabel('Meas - LSTM Eucl loss')
   plt.savefig('Meas_LSTM_vs_Layer.png')
   #if len(np.where(kf_lstm_diff > 0)):
   ii_lst = [28,36]
   #for ii in np.where(kf_lstm_diff > 0)[0]:
   for ii in ii_lst:
       for jj in range(3):
          plt.subplot(int('13'+str(jj+1)))
          pt_d = plt.plot(test[ii,1:,jj],test[ii,1:,np.mod(jj+1,3)],'r+',label='test')
          plt.hold(True)
          line_d = plt.plot(test[ii,1:,jj],test[ii,1:,np.mod(jj+1,3)],'r')
          pt_r = plt.plot(output[ii,:,jj],output[ii,:,np.mod(jj+1,3)],'g*',label='LSTM')
          line_r = plt.plot(output[ii,:,jj],output[ii,:,np.mod(jj+1,3)],'g')
          pt_kf = plt.plot(filtered[ii,1:,jj],filtered[ii,1:,np.mod(jj+1,3)],'b^',label='KF')
          line_kf = plt.plot(filtered[ii,1:,jj],filtered[ii,1:,np.mod(jj+1,3)],'b')
          #plt.legend(loc='upper right')
          plt.locator_params(nbins=10)
          delta = 0.03
          plt.axis([test[ii,1:,jj].min()+ (-delta) ,test[ii,1:,jj].max() + delta ,test[ii,1:,np.mod(jj+1,3)].min() - delta ,test[ii,1:,np.mod(jj+1,3)].max() + delta])
          if jj == 0:
            plt.xlabel('R Phi vs Z')
          elif jj == 1:
            plt.xlabel('Z vs R')
          else:
            plt.xlabel('R vs R Phi')
       plt.savefig('examples'+str(ii)+'.png')
    #else:
    #  print("KF kick yo butt!")
