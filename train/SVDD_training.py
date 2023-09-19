### Define the center of the hypersphere with the ouput of the untrained SVDD for one noise sample:
sample=validation_dataset[0]
sample=sample.to(device)
center=model(sample)

c=center.reshape(1, 2, 2008)
c2=center.reshape(1, 2, 2008)

center=center.reshape(1, 2, 2008)

for i in range(t_batch_size-1):
  c=torch.cat((c, center),0)

c=c.detach()
c.requires_grad=False

for j in range(v_batch_size-1):
  c2=torch.cat((c2, center),0)
c2=c2.detach()
c2.requires_grad=False

n_epochs=150
valid_len=400
batch_size=t_batch_size

while epoch < n_epochs:

    train_loss = 0.0
    valid_loss = 0.0
    sbcnt=0
    vbatch=0
    model.train()
    for batch_indx, data in enumerate(training_dataloader):
        print("epoch:", epoch, "train batch:", batch_indx)
        data= data.to(device).float()
        optimizer.zero_grad()
        output=model(data)
        loss=F.mse_loss(output, c)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*data.size(0)
        train_Loss_list2.append(loss.item())
        sbcnt+=1
        #validation:
        if (batch_indx+1)%valid_len==0:
            model.eval()
            vbatch=0
            for data in validation_dataloader:
                data = data.to(device).float()
                output=model(data)
                loss = F.mse_loss(output, c2)
                valid_loss += loss.item()*data.size(0)
                vbatch+=1
            model.train()
            #Calculate losses:
            train_loss=train_loss/(valid_len*batch_size)
            valid_loss = valid_loss / len(validation_dataloader.dataset)
            print("train loss:", train_loss, flush=True)
            print("valid loss:", valid_loss, flush=True)

            #curve data:
            train_Loss_list.append(train_loss)
            valid_Loss_list.append(valid_loss)
            pd.DataFrame(train_Loss_list).to_csv('/content/drive/MyDrive/Anomaly_detection_GW/Results/Loss_lists/SVDD_tloss.csv')
            pd.DataFrame(valid_Loss_list).to_csv('/content/drive/MyDrive/Anomaly_detection_GW/Results/Loss_lists/SVDD_vloss.csv')
            checkpoint_manager.step(metric=valid_loss, epoch=epoch)
            train_loss=0.0
            valid_loss=0.0
    plt.plot(range(len(train_Loss_list)), train_Loss_list, label='train loss')
    plt.plot(range(len(valid_Loss_list)), valid_Loss_list, label='valid loss')
    plt.legend()
    plt.show()
    plt.plot(range(len(valid_Loss_list)), valid_Loss_list, label='valid loss')
    plt.legend()
    plt.show()
    print("total train batches:", sbcnt)
    #print(data, target)
    lr_sched.step()

    print("total validation batches:",vbatch)
    epoch+=1
