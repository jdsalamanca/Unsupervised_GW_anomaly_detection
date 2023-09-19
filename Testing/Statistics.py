r=(0.5,1.5)
threshold=r[0]
TPR_list=[]
FPR_list=[]
config=get_config()
results_I=[]
results_N=[]

#We calculate the True Positive Rate (TPR) and the False Positive Rate (FPR) for each SNR bin:
# We do this running through many different thresholds of MSE
with h5py.File(config['data']['testing'], 'r') as snr_file:
  inj_snr = np.array(snr_file['/injection_parameters/injection_snr'])
  with h5py.File(predictions_file_path, 'r') as loss_file:
    m_I_Loss=np.array(loss_file['Injection'])
    m_N_Loss=np.array(loss_file['Noise'])

    while threshold<r[1]:
      TP=0
      FP=0
      snr_TP=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      snr_total_P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      for i in range(len(m_I_Loss)):
        snr=round(inj_snr[i])
        snr_total_P[snr-5]+=1
        if m_I_Loss[i]>threshold:
          snr_TP[snr-5]+=1
          TP+=1
      snr_TPR=np.array(snr_TP)/np.array(snr_total_P)
      results_I.append(snr_TPR)
      TPR=TP/len(m_I_Loss)
      TPR_list.append(TPR)
      for j in range(len(m_N_Loss)):
        if m_N_Loss[j]>threshold:
          FP+=1
      FPR=FP/len(m_N_Loss)
      FPR_list.append(FPR)
      threshold+=((r[1]-r[0])/4000)

TPR_list=np.flip(TPR_list)
FPR_list=np.flip(FPR_list)
results_I=np.flip(results_I, axis=0)

# We visualize the Detection Ratio (TPR) vs the Signal to Noise Ratio (SNR) for multiple thresholds of MSE
y=range(5,21)
i=300
while i<350:
  plt.plot(y,results_I[i])
  i+=5
plt.xlabel('SNR')
plt.ylabel('DR')
plt.legend(loc=(1.04, 0))
plt.show()

#Visualize the area under the curve (AUC) of the ROC curves (FPR vs TPR for all the thresholds of MSE):

results_I=np.array(results_I)
areas=[]
for i in range(16):
  y=results_I[:,i]
  #y=np.flip(y)
  area=trapz(y=y, x=FPR_list)
  plt.plot(FPR_list, y, label=f'SNR={i+5}; AUC={round(area, 3)}')
  areas.append(area)
plt.legend(loc=(1.04, 0))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()
print(areas)
