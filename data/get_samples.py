def get_normalization_from_samplefile(file_path:str) ->dict:

  if not os.path.exists(file_path):
    raise FileNotFoundError(f'Sample file "{file_path}" does not exist!')

  with h5py.File(file_path,'r') as hdf_file:

    h1_mean = float(hdf_file['normalization_parameters'].attrs['h1_mean'])
    l1_mean = float(hdf_file['normalization_parameters'].attrs['l1_mean'])
    h1_std = float(hdf_file['normalization_parameters'].attrs['h1_std'])
    l1_std = float(hdf_file['normalization_parameters'].attrs['l1_std'])

    return dict(h1_mean=h1_mean, l1_mean=l1_mean, h1_std=h1_std, l1_std=l1_std)

def get_noise_samples(file_path: str, sample_type: str, normalization: dict, n_samples, sample_z, cropp: bool):
  if not os.path.exists(file_path):
    raise FileNotFoundError(f'Sampel file "{file_path}"does not exist!')

  if normalization is None:
    print("No normalization parameters")
    normalization={'h1_mean':0.0, 'h1_std':1.0,'l1_mean':0.0, 'l1_std':1.0}

  m_h1=normalization['h1_mean']
  std_h1=normalization['h1_std']
  m_l1=normalization['l1_mean']
  std_l1=normalization['l1_std']

  print("Getting samples from", sample_z, "to", n_samples)

  if sample_type=='noise':
    with h5py.File(file_path,'r') as hdf_file:
      samples=np.dstack([(np.array(hdf_file['/noise_samples/h1_strain'][sample_z:n_samples])- m_h1)/std_h1, (np.array(hdf_file['/noise_samples/l1_strain'][sample_z:n_samples])-m_l1)/std_l1])
      print("stacking done!")
      print("swaping axes..")
      samples=np.swapaxes(samples,1,2)
  else:
    with h5py.File(file_path,'r') as hdf_file:
      samples=np.dstack([(np.array(hdf_file['/injection_samples/h1_strain'][sample_z:n_samples])- m_h1)/std_h1, (np.array(hdf_file['/injection_samples/l1_strain'][sample_z:n_samples])-m_l1)/std_l1])
      print("stacking done!")
      print("swaping axes..")
      samples=np.swapaxes(samples,1,2)

  if cropp==True:
    samples=samples[:,:,500:-500]

  return samples
