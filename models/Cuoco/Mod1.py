class CuocoAE2(nn.Module):
    def __init__(self, in_ch: int):
        super(CuocoAE2, self).__init__()
        self.conv1=nn.Conv1d(in_ch, 256, 3)
        self.conv11=nn.Conv1d(256, 256, 20)
        self.max1=nn.MaxPool1d(2)
        self.conv2=nn.Conv1d(256, 128, 3)
        self.conv22=nn.Conv1d(128, 128, 20)
        self.up1=nn.Upsample(scale_factor=2)
        self.conv3=nn.Conv1d(128, 256, 3)
        self.conv33=nn.Conv1d(256, 256, 20)
        self.out=nn.ConvTranspose1d(256,2,86)

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv11(x)
        x=F.relu(x)
        x=self.max1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.conv22(x)
        x=F.relu(x)
        x=self.up1(x)
        x=F.relu(x)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv33(x)
        x=F.relu(x)
        x=self.out(x)
        x=F.tanh(x)
        return x