class SegmentedVAE(nn.Module):
    def __init__(self, in_ch: int):
        super(SegmentedVAE,self).__init__()

        ##Encoder:
        self.Encoder=nn.Sequential(nn.Conv1d(in_ch,128,3),
            nn.ReLU(),
            nn.Conv1d(128,128,20),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(128,128,3),
            nn.ReLU(),
            nn.Conv1d(128,128,20),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(128,128,3),
            nn.ReLU(),
            nn.Conv1d(128,128,20))


        ##Gaussian distribution:

        self.mu=nn.Linear(4059,4059)
        self.sigma=nn.Linear(4059,4059)
        self.D=torch.distributions.Normal(0, 1)
        self.D.loc=self.D.loc.cuda()
        self.D.scale=self.D.scale.cuda()
        self.KL=0
        self.m=0
        self.s=0
        self.sample=0

        ##Decoder:
        self.Decoder=nn.Sequential(nn.ConvTranspose1d(128,128,3),
            nn.ReLU(),
            nn.ConvTranspose1d(128,128,20),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128,128,3),
            nn.ReLU(),
            nn.ConvTranspose1d(128,128,20),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128,128,3),
            nn.ReLU(),
            nn.ConvTranspose1d(128,2,21),
            nn.Tanh())


    def forward(self,x):

        x=self.Encoder(x)

        #Reparametrization:
        self.s=self.sigma(x)
        self.m=self.mu(x)
        self.sample=self.D.sample(self.s.shape)
        z=self.m+self.s*self.sample

        #Caluclate the KLD:
        mu2_m=torch.mean(self.m**2)
        sigma2_m=torch.mean(self.s**2)
        self.KL=mu2_m+sigma2_m-torch.log(sigma2_m)-1

        x=self.Decoder(z)

        return x
