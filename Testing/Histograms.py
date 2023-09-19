###-------------------------------------------------------------
#                     VISUALIZE RESULTS
###-------------------------------------------------------------

#First we get the results from the file:

predictions_file_name = 'MSE_AE.hdf'
results_dir='/content/drive/MyDrive/Anomaly_detection_GW/Results'
predictions_file_path = os.path.join(results_dir, predictions_file_name)
snr_losses=[]
config=get_config()
a=0

#We create an array for the MSE loss for each SNR value (5-20):
while a<16:
  snr_losses.append([])
  a+=1
print(snr_losses)
with h5py.File(predictions_file_path, 'r') as loss_file:
  m_I_Loss=np.array(loss_file['Injection'])
  m_N_Loss=np.array(loss_file['Noise'])
  with h5py.File(config['data']['testing'], 'r') as snr_file:
    inj_snr = np.array(snr_file['/injection_parameters/injection_snr'])
    l=m_I_Loss.shape[0]
    print(l)
    for i in range(l):
      snr=round(inj_snr[i])
      snr_losses[snr-5].append(m_I_Loss[i])

#Visualize the scores for each SNR bin

r=(0.5,1.5)
for j in range(16):
  plt.hist(snr_losses[j], bins=300, range=r, label=f'SNR={j+5}')

#plt.hist(m_N_loss, bins=200, range=(0,25), label='Noise')
plt.xlabel('MSE')
plt.ylabel('Frecuency')
plt.legend()
plt.show()

#Visualize the difference of the scores between the noise and the injection samples:

plt.hist(m_N_Loss, bins=300, label ='Loss for Noise samples')
plt.hist(m_I_Loss, bins=300, label= 'Loss for Injection samples')
plt.legend()
plt.show()
plt.hist(m_N_Loss, bins=300, range=r, label ='Loss for Noise samples')
plt.hist(m_I_Loss, bins=300, range=r, label= 'Loss for Injection samples')
plt.legend()
plt.show()
