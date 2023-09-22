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
    mu=model.m(data)
    n_rows=output.size(dim=0)
    n_cols=output.size(dim=2)
    V_loss=[]

    for i in range(n_rows):
        Losses['noise'].append(mu)

    batch+=1
    print("Batch:", batch, flush=True)

for data in injection_dataloader:

    #data=data.reshape(-1,1,16384)
    #target= target.reshape(-1,1,16384)
    data = data.to(device).float()
    #data, target= data.float(), target.float()
    mu=model.m(data)
    n_rows=output.size(dim=0)
    n_cols=output.size(dim=2)
    V_loss=[]

    for i in range(n_rows):
        Losses['noise'].append(mu)

    batch+=1
    print("Batch:", batch, flush=True)
