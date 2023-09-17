class SLowDimVAE(nn.Module):
    def __init__(self, in_ch: int):
        super(SLowDimVAE, self).__init__()

        ##Encoder:
        self.conv1=nn.Conv1d(in_ch,64,7)
        self.max1=nn.MaxPool1d(4,4)
        self.conv2=nn.Conv1d(64,128,9)
        self.max2=nn.MaxPool1d(4,4)
        self.conv3=nn.Conv1d(128,256,8)
        self.max3=nn.MaxPool1d(4,4)
        self.conv4=nn.Conv1d(256,256,8)


        ##Gaussian distribution:
        s=246
        self.mu=nn.Linear(s,s)
        self.sigma=nn.Linear(s,s)
        self.D=torch.distributions.Normal(0, 1)
        self.D.loc=self.D.loc.cuda()
        self.D.scale=self.D.scale.cuda()
        self.KL=0
        self.z=0

        ##Decoder:
        self.deconv1=nn.ConvTranspose1d(256,256,8)
        self.up1=nn.Upsample(scale_factor=4)
        self.deconv2=nn.ConvTranspose1d(256,128,10)
        self.up2=nn.Upsample(scale_factor=4)
        self.deconv3=nn.ConvTranspose1d(128,64,11)
        self.up3=nn.Upsample(scale_factor=4)
        self.deconv4=nn.ConvTranspose1d(64,2,9)



    def forward(self,x):
        ## Encoder:
        x=self.conv1(x)
        x=F.leaky_relu(x)
        x=self.max1(x)
        x=self.conv2(x)
        x=F.leaky_relu(x)
        x=self.max2(x)
        x=self.conv3(x)
        x=F.leaky_relu(x)
        x=self.max3(x)
        x=self.conv4(x)
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
        z=F.leaky_relu(x)
        x=self.up1(x)
        x=self.deconv2(x)
        x=F.leaky_relu(x)
        x=self.up2(x)
        x=self.deconv3(x)
        x=F.leaky_relu(x)
        x=self.up3(x)
        x=self.deconv4(x)
        x=F.tanh(x)
        return x
