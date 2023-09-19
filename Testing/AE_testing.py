####------------------------------------------------------------------------
#                          TEST THE AE WITH NOISE
####------------------------------------------------------------------------
Losses={'injection': list(), 'noise': list()}

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
        l=F.mse_loss(output[i], data[i])
        loss_h1 = (output[i,0]-data[i,0]).cpu().detach().numpy()
        loss_l1 = (output[i,1]-data[i,1]).cpu().detach().numpy()
        loss=(loss_h1+loss_l1)/2
        N_batch_ind_loss.append(loss**2)
        V_loss.append(l.item())
        m_N_loss.append(l.item())
        Losses['noise'].append(l.item())

    batch+=1
    print("Batch:", batch, flush=True)
    #print("Loss of the last v batch:", V_loss)
    col_list=range(n_cols)

    if batch%20==0:
        while q<u:
            plt.scatter(col_list, N_batch_ind_loss[q])
            plt.show()
            q+=1
        u+=3


###-------------------------------------------------------------------
#                  TEST THE AE WITH INJECTIONS
###-------------------------------------------------------------------

batch=0
I_batch_ind_loss=[]
m_I_loss=[]
q=0
u=3
for data in injection_dataloader:
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
        l=F.mse_loss(output[i], data[i])
        loss_h1 = (output[i,0]-data[i,0]).cpu().detach().numpy()
        loss_l1 = (output[i,1]-data[i,1]).cpu().detach().numpy()
        loss=(loss_h1+loss_l1)/2
        I_batch_ind_loss.append(loss**2)
        m_I_loss.append(l.item())
        V_loss.append(l.item())
        Losses['injection'].append(l.item())

    batch+=1
    print("Batch:", batch)
    #print("Loss of the last v batch:", V_loss)
    col_list=range(n_cols)
    if batch%20==0:
        while q < u:
            plt.scatter(col_list, I_batch_ind_loss[q])
            plt.show()
            q+=1
        u+=3

###----------------------------------------------------------
#                     SAVE THE RESULTS
###----------------------------------------------------------

predictions_file_name = 'MSE_AE.hdf'
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
    hdf_file.create_dataset(name='Noise', data=Losses['noise'])
    hdf_file.create_dataset(name='Injection', data=Losses['injection'])
print("hdf file done!", flush=True)
