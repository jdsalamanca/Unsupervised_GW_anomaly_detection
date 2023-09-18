###------------------------------------------------------------
#                     MAIN CODE
###------------------------------------------------------------


valid_len=3000
n_epochs=200

while epoch<n_epochs:
    train_loss=0.0
    valid_loss=0.0
    model.train()
    for batch_indx, data in enumerate(training_dataloader):
        data=data.to(device).float()
        optimizer.zero_grad()
        output=model(data)
        loss=F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        print("epoch:" ,epoch, "training batch: ", batch_indx)
        if (batch_indx+1)%valid_len==0:
            model.eval()
            avrg_t_loss=train_loss/valid_len
            train_loss_list.append(avrg_t_loss)
            for data in validation_dataloader:
                data=data.to(device).float()
                output=model(data)
                loss=F.mse_loss(output, data)
                valid_loss+=loss.item()
            avrg_v_loss=valid_loss/n_v_batches
            valid_loss_list.append(avrg_v_loss)
            checkpoint_manager.step(metric=avrg_v_loss, epoch=epoch)
            model.train()
            print("Train loss:", avrg_t_loss)
            print("Valid loss:", avrg_v_loss)
            train_loss=0.0
            valid_loss=0.0
            pd.DataFrame(train_loss_list).to_csv('/content/drive/MyDrive/Anomaly_detection_GW/Results/Loss_lists/AE_new_data_normalized_t_loss.csv')
            pd.DataFrame(valid_loss_list).to_csv('/content/drive/MyDrive/Anomaly_detection_GW/Results/Loss_lists/SE_new_data_normalized_vloss.csv')
    validations=len(train_loss_list)
    plt.plot(range(validations), train_loss_list, label='train loss')
    plt.plot(range(validations), valid_loss_list, label='valid loss')
    plt.legend()
    plt.show()
    plt.plot(range(validations), valid_loss_list, label='valid loss')
    plt.legend()
    plt.show()
    epoch+=1
    lr_sched.step()
