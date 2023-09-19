# Vector REconstruction Error technique as performed by Torabi et al.

Losses={'injection': list(), 'noise': list()}
Individual_Losses={'injection': list(), 'noise': list()}

model.eval()
batch=0
N_batch_ind_loss=[]
m_N_loss=[]
q=0
u=3
for data in noise_dataloader:

    #data=data.reshape(-1,1,16384)
    #target= target.reshape(-1,1,16384)
    data = data.to(device).float()
    #data, target= data.float(), target.float()
    output=model(data)
    n_rows=output.size(dim=0)
    n_cols=output.size(dim=2)
    V_loss=[]

    for i in range(n_rows):
        ind_loss=[]
        #-------------------------------------------------
        #Lets leave this stupid mistake to remember it:
        #l=F.mse_loss(output[i,0], data[i,0])
        #-------------------------------------------------
        l=F.mse_loss(output[i], data[i])
        #-----------------------------------------------------------------------------------------------------------
        #The followinf is actually the squared mean error, not equivalent to the MSE function provided by pytorch:
        ind_loss_h1 = (output[i,0]-data[i,0]).cpu().detach().numpy()
        ind_loss_l1 = (output[i,1]-data[i,1]).cpu().detach().numpy()
        ind_loss=(ind_loss_h1+ind_loss_l1)/2
        N_batch_ind_loss.append(ind_loss**2)
        #-----------------------------------------------------------------------------------------------------------

        #Lets calculate the real individual squared error to obtain an array of erros scores for each strain of data:
        loss_h1 = ((output[i,0]-data[i,0])**2).cpu().detach().numpy()
        loss_l1 = ((output[i,1]-data[i,1])**2).cpu().detach().numpy()
        loss=(loss_h1+loss_l1)/2
        V_loss.append(l.item())
        m_N_loss.append(l.item())
        Losses['noise'].append(l.item())
        Individual_Losses['noise'].append(loss)

    batch+=1
    print("Batch:", batch, flush=True)
    #print("Loss of the last v batch:", V_loss)
    col_list=range(n_cols)

    if batch%20==0:
        while q<u:
            plt.scatter(col_list, N_batch_ind_loss[q], label='SME')
            plt.legend()
            plt.show()
            plt.scatter(col_list, Individual_Losses['noise'][q], label='MSE')
            plt.legend()
            plt.show()
            q+=1
        u+=3

  batch=0
I_batch_ind_loss=[]
m_I_loss=[]
q=0
u=3
total_rows=0
print('Size of the dataloader:', len(injection_dataloader.dataset))
for data in injection_dataloader:
    #data=data.reshape(-1,1,16384)
    #target= target.reshape(-1,1,16384)
    data = data.to(device).float()
    #data, target= data.float(), target.float()
    output=model(data)
    n_rows=output.size(dim=0)
    total_rows+=n_rows
    print('Number of rows for this batch:' , n_rows)
    print('Current row count:', total_rows)
    n_cols=output.size(dim=2)
    V_loss=[]
    for i in range(n_rows):
        ind_loss=[]
        l=F.mse_loss(output[i], data[i])

        ind_loss_h1 = (output[i,0]-data[i,0]).cpu().detach().numpy()
        ind_loss_l1 = (output[i,1]-data[i,1]).cpu().detach().numpy()
        ind_loss=(ind_loss_h1+ind_loss_l1)/2
        I_batch_ind_loss.append(ind_loss**2)
        # Calculate the real individual squared error:

        loss_h1 = ((output[i,0]-data[i,0])**2).cpu().detach().numpy()
        loss_l1 = ((output[i,1]-data[i,1])**2).cpu().detach().numpy()
        loss=(loss_h1+loss_l1)/2

        m_I_loss.append(l.item())
        V_loss.append(l.item())

        Losses['injection'].append(l.item())
        Individual_Losses['injection'].append(loss)
    batch+=1
    print("Batch:", batch)
    #print("Loss of the last v batch:", V_loss)
    col_list=range(n_cols)
    if batch%20==0:
        while q < u:
            plt.scatter(col_list, I_batch_ind_loss[q], label='SME')
            plt.legend()
            plt.show()
            plt.scatter(col_list, Individual_Losses['injection'][q], label='MSE')
            plt.legend()
            plt.show()
            q+=1
        u+=3

  predictions_file_name = 'Ind_MSE_Cuoco_AE_extra_size20_layers_new_data_test_normalized_tanh.hdf'
results_dir='/content/drive/MyDrive/Anomaly_detection_GW/Results'
predictions_file_path = os.path.join(results_dir, predictions_file_name)
print("Creating hdf loss file...", flush=True)
with h5py.File(predictions_file_path, 'a') as hdf_file:

    # Delete the dataset if it already exists
    if 'injection' in hdf_file.keys():
        del hdf_file['injection']
    if 'noise' in hdf_file.keys():
        del hdf_file['noise']

    # Create a new dataset holding the predictions
    hdf_file.create_dataset(name='Noise', data=Individual_Losses['noise'])
    hdf_file.create_dataset(name='Injection', data=Individual_Losses['injection'])
print("hdf file done!", flush=True)

predictions_file_name = 'Ind_MSE_Cuoco_AE_extra_size20_layers_new_data_test_normalized_tanh.hdf'
results_dir='/content/drive/MyDrive/Anomaly_detection_GW/Results'
RE_file_path = os.path.join(results_dir, predictions_file_name)
snr_ind_losses=[]
config=get_config()
a=0
while a<16:
  snr_ind_losses.append([])
  a+=1
print(snr_ind_losses)
with h5py.File(RE_file_path, 'r') as loss_file:
  Ind_I_Loss=np.array(loss_file['Injection'])
  Ind_N_Loss=np.array(loss_file['Noise'])
  with h5py.File(config['data']['testing'], 'r') as snr_file:
    inj_snr = np.array(snr_file['/injection_parameters/injection_snr'])
    l=Ind_I_Loss.shape[0]
    print(l)
    for i in range(l):
      snr=round(inj_snr[i])
      snr_ind_losses[snr-5].append(Ind_I_Loss[i])

R=[0.1, 1.5]
ones=np.ones(16384)
threshold=R[0]
Ind_TPR_list=[]
Ind_FPR_list=[]
Ind_results_I=[]

with h5py.File(config['data']['testing'], 'r') as snr_file:
  inj_snr = np.array(snr_file['/injection_parameters/injection_snr'])
  with h5py.File(RE_file_path, 'r') as loss_file:
    Ind_I_Loss=np.array(loss_file['Injection'])
    Ind_N_Loss=np.array(loss_file['Noise'])

    while threshold<R[1]:
      TP=0
      FP=0
      #Keep track of each recovered anomaly on their respective SNR bin:
      snr_TP=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      #Keep track of each ground truth anomaly on their respective SNR bin:
      snr_total_P=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      #Turn the threshold into an array so we can compare each value of the array of RE to the threshold without having another loop to run through the tensor.
      thr=ones*threshold
      for i in range(len(Ind_I_Loss)):
        #Get the injection SNR for the RE we'll process
        snr=round(inj_snr[i])
        #Since this we're on the list of injection samples we know all of them are positives
        snr_total_P[snr-5]+=1
        # Now we get the scores from the current threshold. If only one of the values for each array is over the threshold we'll count it as an anomaly.
        Bool=thr>Ind_I_Loss[i]
        binary=Bool*1
        score=np.prod(binary)
        #If only one RE is below the threshold we'll have a zero in the binary tensor and therfore, the product will be zero. A zero score represents an anomaly
        if score==0:
          snr_TP[snr-5]+=1
          TP+=1
      # We divide the recovered signals (TP) by the total number of signals for each SNR value (ordered by the place in the first dimension of the array)
      # to get the TPR for each SNR value
      snr_TPR=np.array(snr_TP)/np.array(snr_total_P)
      Ind_results_I.append(snr_TPR)
      # We get the general TPR for the threshold
      TPR=TP/len(m_I_Loss)
      Ind_TPR_list.append(TPR)
      for j in range(len(Ind_N_Loss)):
      # We keep track of all false positive
        Bool=thr>Ind_N_Loss[j]
        binary=Bool*1
        score=np.prod(binary)
        # Same logic, a zero score counts as an anomaly. Since we are on the noise samples, each detected anomaly is a false positive
        if score==0:
          FP+=1
      # The whole list is noise so the FPR is just the number of FP divided by the size of the list
      FPR=FP/len(m_N_Loss)
      Ind_FPR_list.append(FPR)
      threshold+=((R[1]-R[0])/1000)
      print(threshold)
