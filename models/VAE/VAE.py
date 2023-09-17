class VAE(nn.Module):
    def __init__(self, in_ch: int):
        super(VAE,self).__init__()

        ##Encoder:
        self.conv1=nn.Conv1d(in_ch,64,10)
        self.conv11=nn.Conv1d(64,64,8)
        self.max1=nn.MaxPool1d(2,2)
        self.conv2=nn.Conv1d(64,128,8)
        self.conv22=nn.Conv1d(128,128,8)
        self.max2=nn.MaxPool1d(2,2)
        self.conv3=nn.Conv1d(128,256,8)
        self.conv33=nn.Conv1d(256,256,8)

        ##Gaussian distribution:
        self.mu=nn.Linear(4071,4071)
        self.sigma=nn.Linear(4071,4071)
        self.D=torch.distributions.Normal(0, 1)
        self.D.loc=self.D.loc.cuda()
        self.D.scale=self.D.scale.cuda()
        self.KL=0
        self.z=0

        ##Decoder:
        self.deconv1=nn.ConvTranspose1d(256,128,8)
        self.deconv11=nn.ConvTranspose1d(128,128,8)
        self.up1=nn.Upsample(scale_factor=2)
        self.deconv2=nn.ConvTranspose1d(128,128,8)
        self.deconv22=nn.ConvTranspose1d(128,64,8)
        self.up2=nn.Upsample(scale_factor=2)
        self.deconv3=nn.ConvTranspose1d(64,64,8)
        self.deconv33=nn.ConvTranspose1d(64,2,10)

    def forward(self,x):
        ## Encoder:
        x=self.conv1(x)
        x=F.leaky_relu(x)
        x=self.conv11(x)
        x=F.leaky_relu(x)
        x_1=x
        x=self.max1(x)
        x=self.conv2(x)
        x=F.leaky_relu(x)
        x=self.conv22(x)
        x=F.leaky_relu(x)
        x_2=x
        x=self.max2(x)
        x=self.conv3(x)
        x=F.leaky_relu(x)
        x=self.conv33(x)
        x=F.leaky_relu(x)
        #print("Encoder output:", x)

        ##Sampling form Gaussian distribution:
        sigma=torch.sigmoid(self.sigma(x))
        mu=torch.sigmoid(self.mu(x))
        sample=self.D.sample(sigma.shape)
        z=mu+sigma*sample
        mu2_m=torch.mean(mu**2)
        sigma2_m=torch.mean(sigma**2)
        kl=self.KL=mu2_m+sigma2_m-torch.log(sigma2_m)-1

        ##Decoder:
        x=self.deconv1(z)
        x=F.leaky_relu(x)
        x=self.deconv11(x)
        x=F.leaky_relu(x)
        x=self.up1(x)
        x=self.deconv2(x)
        x=F.leaky_relu(x)
        x=self.deconv22(x)
        x=F.leaky_relu(x)
        x=self.up2(x)
        x=self.deconv3(x)
        x=F.leaky_relu(x)
        x=self.deconv33(x)
        #x=F.leaky_relu(x,negative_slope=0.8)
        x=F.sigmoid(x)
        return x
