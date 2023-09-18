class SVDD(nn.Module):
    def __init__(self, in_ch: int):
        super(SVDD,self).__init__()
        self.maxp=nn.MaxPool1d(2)
        self.conv1=nn.Conv1d(in_ch, 64, 3)
        self.conv11=nn.Conv1d(64, 64, 20)
        self.max=nn.MaxPool1d(2)
        self.conv2=nn.Conv1d(64, 128, 3)
        self.conv22=nn.Conv1d(128, 128, 20)
        self.conv3=nn.Conv1d(128, 128, 3)
        self.conv33=nn.Conv1d(128, 64, 20)
        self.conv4=nn.Conv1d(64, 64, 3)
        self.conv44=nn.Conv1d(64, 2, 20)


    def forward(self,x):

        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv11(x)
        x=F.relu(x)
        x=self.maxp(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.conv22(x)
        x=F.relu(x)
        x=self.maxp(x)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv33(x)
        x=F.relu(x)
        x=self.maxp(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv44(x)
        x=F.relu(x)


        return x
