class NoiseDataset(torch.utils.data.Dataset):
  def __init__(self, mode: str, sample_type: str, cropp: bool):

    if mode not in ("training", "validation", "testing"):
      raise ValueError('mode must be "training", "testing" or "validation"!')

    config=get_config()
    data_paths=config['data']

    normalization= get_normalization_from_samplefile(data_paths[mode])
    print("Normalization parameters", normalization)
    if mode=='training':
      n_samples=188000
      sample_z=0
      print(f"getting '{n_samples-sample_z}' noise training samples..", flush=True)
    if mode== 'validation':
      n_samples=10000
      sample_z=0
      print(f"getting '{n_samples-sample_z}' noise validation samples..", flush=True)
    if mode=='testing' and sample_type=='noise':
      n_samples=25000
      sample_z=0
      print(f"getting '{n_samples-sample_z}' noise testing samples..", flush=True)
    if mode=='testing' and sample_type=='injection':
      n_samples=25000
      sample_z=0
      print(f"getting '{n_samples-sample_z}' injection testing samples..", flush=True)


    self.data= get_noise_samples(file_path=data_paths[mode], sample_type=sample_type, normalization=normalization, n_samples=n_samples, sample_z=sample_z, cropp=cropp)
    print("...samples loaded")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):

    data=torch.tensor(self.data[index]).float()
    return data
