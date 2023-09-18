###---------------------------------------------------
#            COPIED TRAIN LOOP
###---------------------------------------------------
n_epochs=150
valid_len=400
batch_size=t_batch_size
valid_Loss_list2=[]
train_Loss_list2=[]
alpha=1
beta=1

while epoch < n_epochs:

    train_loss = 0.0
    valid_loss = 0.0
    valid_loss2 = 0.0
    correct = 0.0
    sbcnt=0
    model.train()
    for batch_indx, data in enumerate(training_dataloader):
        data= data.to(device).float()
        optimizer.zero_grad()
        output=model(data)
        loss=alpha*F.mse_loss(output, data)+beta*model.KL
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*data.size(0)
        train_Loss_list2.append(loss.item())
        print("epoch:", epoch, "train batch:", batch_indx)
        sbcnt+=1
        #validation:
        if (batch_indx+1)%valid_len==0:
            model.eval()
            vbatch=0
            for data in validation_dataloader:

                data = data.to(device).float()
                output=model(data)
                loss = alpha*F.mse_loss(output, data)+ beta*model.KL
                loss2= alpha*F.mse_loss(output, data)
                valid_loss += loss.item()*data.size(0)
                valid_loss2 += loss2.item()*data.size(0)
                vbatch+=1
            model.train()
            #Calculate losses:
            train_loss=train_loss/(valid_len*batch_size)
            valid_loss = valid_loss / len(validation_dataloader.dataset)
            valid_loss2 = valid_loss2 / len(validation_dataloader.dataset)
            print("train loss:", train_loss, flush=True)
            print("valid loss:", valid_loss, flush=True)
            print("valid loss:", valid_loss2, flush=True)
            #curve data:
            train_Loss_list.append(train_loss)
            valid_Loss_list.append(valid_loss)
            valid_Loss_list2.append(valid_loss2)
            pd.DataFrame(train_Loss_list).to_csv('/content/drive/MyDrive/Anomaly_detection_GW/Results/Loss_lists/SegmentedVAE_no_transpose_tloss.csv')
            pd.DataFrame(valid_Loss_list).to_csv('/content/drive/MyDrive/Anomaly_detection_GW/Results/Loss_lists/SegmentedVAE_no_transpose_vloss.csv')
            checkpoint_manager.step(metric=valid_loss2, epoch=epoch)
            train_loss=0.0
            valid_loss=0.0
            valid_loss2= 0.0
    plt.plot(range(len(train_Loss_list)), train_Loss_list, label='train loss')
    plt.plot(range(len(valid_Loss_list)), valid_Loss_list, label='valid loss')
    plt.legend()
    plt.show()
    plt.plot(range(len(valid_Loss_list)), valid_Loss_list, label='valid loss')
    plt.legend()
    plt.show()
    plt.plot(range(len(valid_Loss_list2)), valid_Loss_list2, label='valid loss')
    plt.legend()
    plt.show()
    plt.plot(range(len(train_Loss_list2)), train_Loss_list2, label='train loss')
    plt.legend()
    plt.show()
    print("total train batches:", sbcnt)
    #print(data, target)
    lr_sched.step()

    print("total validation batches:",vbatch)
    epoch+=1
