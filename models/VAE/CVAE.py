class CVAE(nn.Module):
    def __init__(self, in_ch: int):
        super(CVAE,self).__init__()
        st_ch=128
        ##Encoder:
        self.conv1=nn.Conv1d(in_ch,st_ch,10)
        self.conv2=nn.Conv1d(st_ch,st_ch,6)
        self.max1=nn.MaxPool1d(2,1)
        self.conv3=nn.Conv1d(st_ch,st_ch*2,6)
        self.conv4=nn.Conv1d(st_ch*2,st_ch*2,6)


        ##Gaussian distribution:
        self.mu1=nn.Conv1d(st_ch*2, st_ch*2, 6, padding=2)
        self.mu2=nn.Conv1d(st_ch*2, st_ch*2, 6, padding=3)
        self.mu=nn.Conv1d(st_ch*2, st_ch*4, 6, padding=3)
        self.sigma1=nn.Conv1d(st_ch*2, st_ch*2, 6, padding=2)
        self.sigma2=nn.Conv1d(st_ch*2, st_ch*2, 6, padding=3)
        self.sigma=nn.Conv1d(st_ch*2, st_ch*4, 6, padding=3)
        self.D=torch.distributions.Normal(0, 1)
        self.D.loc=self.D.loc.cuda()
        self.D.scale=self.D.scale.cuda()
        self.KL=0


        ##Decoder:

        self.deconv1=nn.ConvTranspose1d(st_ch*4,st_ch*4,6)
        self.deconv2=nn.ConvTranspose1d(st_ch*4,st_ch*2,6)
        self.deconv3=nn.ConvTranspose1d(st_ch*2,st_ch*2,6)
        self.deconv4=nn.ConvTranspose1d(st_ch*2,in_ch,10)


    def forward(self,x):
        ## Encoder:
        x=self.conv1(x)
        x=F.leaky_relu(x)
        x=self.conv2(x)
        x=F.leaky_relu(x)
        x=self.max1(x)
        x=self.conv3(x)
        x=F.leaky_relu(x)
        x=self.conv4(x)
        x=F.leaky_relu(x)

	      #Sampling form Gaussian distribution:
        s=self.sigma1(x)
        s=self.sigma2(s)
        sigma=torch.sigmoid(self.sigma(s))
        m=self.mu1(x)
        m=self.mu2(m)
        mu=torch.sigmoid(self.mu(m))
        sample=self.D.sample(sigma.shape)
        z=mu+sigma*sample
        mu2_m=torch.mean(mu**2)
        sigma2_m=torch.mean(sigma**2)
        kl=self.KL=mu2_m+sigma2_m-torch.log(sigma2_m)-1
        if kl>1000000:
            print("kl:", kl)
            mu_s=mu**2
            sigma_s=sigma**2
            print("max value of mu:", torch.max(mu))
            print("max value of sigma:", torch.max(sigma))
            print("max value of mu**2:", torch.max(mu_s))
            print("max value of sigma**2:", torch.max(sigma_s))
            print("mean value of mu**2:", torch.mean(mu_s))
            print("mean value of sigma**2:", torch.mean(sigma_s))
            print("mean value of log sigma**2:", torch.mean(torch.log(sigma_s)))
            print("min value of sigma**2:", torch.min(sigma_s))
            print("min value of log sigma**2:", torch.min(torch.log(sigma_s)))

        ##Decoder:

        x=self.deconv1(z)
        x=F.leaky_relu(x)
        x=self.deconv2(x)
        x=F.leaky_relu(x)
        x=self.deconv3(x)
        x=F.leaky_relu(x)
        x=self.deconv4(x)
        x=F.leaky_relu(x,negative_slope=0.8)
        return x
