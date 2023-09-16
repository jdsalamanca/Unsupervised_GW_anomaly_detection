class CheckpointManager(object):

  def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, checkpoints_directory: str= '/content/drive/MyDrive/Anomaly_detection_GW/Autoencoder_checkpoints', mode: str= 'min,', step_size: int = 1, last_epoch: int =-1, verbose: bool=True):
    if not isinstance(model, torch.nn.Module):
      raise TypeError(f'{type(model).__name__} is not a Module')

    if not isinstance(optimizer, torch.optim.Optimizer):
      raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')

    if not scheduler.__module__ == 'torch.optim.lr_scheduler':
      raise TypeError(f'{type(scheduler).__name__} is not a Scheduler')

    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.checkpoints_directory = checkpoints_directory
    self.mode = mode
    self.step_size = step_size
    self.last_epoch = last_epoch
    self.verbose = verbose

    self.initialize_checkpoints_directory()
    self.best_metric = np.inf * (-1 if mode == 'max' else 1)

  def initialize_checkpoints_directory(self):

    Path(self.checkpoints_directory).mkdir(parents=True, exist_ok=True)

  def step(self, metric: float, epoch: int = None):

    if not epoch:
      epoch = self.last_epoch + 1
    self.last_epoch=epoch

      ##Save the best model checkpoint if applicable

    if(self.mode=="min" and metric<self.best_metric) or (self.mode=="max" and metric>self.best_metric):
      self.best_metric=metric
      self.save_checkpoint(checkpoint=self.get_current_checkpoint(), name='best_Cuoco_AE_extra_size2_leaky_normalized_tanh.pth')

    ##Save regular checkpoint every stepsize epochs:

    if(self.step_size>0) and (self.last_epoch % self.step_size==0):
      self.save_checkpoint(checkpoint=self.get_current_checkpoint(), name=f'best_VAE_256_3pool_800kparams{epoch}.pth')

  def save_checkpoint(self, checkpoint: dict, name: str):
    torch.save(checkpoint, os.path.join(self.checkpoints_directory, name))

    if self.verbose:
      print(f'Saved checkpoint: {name}')

  def load_checkpoint(self, checkpoint_file_path: str):
    if not os.path.exists(checkpoint_file_path):
      raise FileNotFoundError(f'{checkpoint_file_path} does not exist')

    checkpoint= torch.load(checkpoint_file_path)
    print("...checkpoint file loaded")

    self.model.load_state_dict(checkpoint['model_state_dict'])
    print("...model loaded")
    self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
    print("...optimizer loaded")
    self.scheduler.load_state_dict(checkpoint['sched_state_dict'])
    print("...scheduler loaded")

    self.last_epoch=checkpoint['last_epoch']
    self.best_metric=checkpoint['best_metric']

  def get_current_checkpoint(self):
    if isinstance(self.model, torch.nn.DataParallel):
      model_state_dict= self.model.module.state_dict()
    else:
      model_state_dict= self.model.state_dict()

    checkpoint= dict(model_state_dict=model_state_dict, optim_state_dict=self.optimizer.state_dict(), sched_state_dict=self.scheduler.state_dict(), best_metric=self.best_metric, last_epoch=self.last_epoch)

    return checkpoint
