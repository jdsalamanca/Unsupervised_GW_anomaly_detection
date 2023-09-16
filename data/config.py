def  get_config() -> dict:
  #The drive path:

  selection=1
  #1 New unsupervised data with good labeling
  if os.path.exists('/content/drive/MyDrive/Anomaly_detection_GW/CONFIG.jason') and selection==1:
    file_path= '/content/drive/MyDrive/Anomaly_detection_GW/CONFIG.jason'
  #2 Old unsupervised data with questionable labeling
  elif os.path.exists('/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_2.jason') and selection==2:
    file_path= '/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_2.jason'
  #3 TFM data with good labeling
  elif os.path.exists('/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_3.jason') and selection==3:
    file_path= '/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_3.jason'
  #4 Gaussian noise data:
  elif os.path.exists('/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_4.jason') and selection==4:
    file_path= '/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_4.jason'
  #5 Low SNR data (1-10)
  elif os.path.exists('/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_lowSNR.jason') and selection==5:
    file_path= '/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_lowSNR.jason'
  #6 Training, validation and testing data from the TFM folder
  elif os.path.exists('/content/drive/MyDrive/TFM/CONFIG.jason') and selection==6:
    file_path= '/content/drive/MyDrive/TFM/CONFIG.jason'
  #Pierini's data:
  elif os.path.exists('/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_Pierini.jason') and selection==7:
    file_path= '/content/drive/MyDrive/Anomaly_detection_GW/CONFIG_Pierini.jason'

  else:
    raise FileNotFoundError('Configuration file not found')

  print("Configuration file path taken:", file_path)
  with open(file_path, 'r') as config_file:
    config = json.load(config_file)

  return config
