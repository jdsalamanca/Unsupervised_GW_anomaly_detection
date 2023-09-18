class BigAutoencoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BigAutoencoderLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder2 = nn.LSTM(hidden_size, 128, batch_first=True)
        self.encoder3 = nn.LSTM(128, 128, batch_first=True)
        self.decoder = nn.LSTM(128, 128, batch_first=True)
        self.decoder2 = nn.LSTM(128, hidden_size, batch_first=True)
        self.decoder3 = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded, _ = self.encoder2(encoded)
        encoded, _ = self.encoder3(encoded)
        decoded, _ = self.decoder(encoded)
        decoded, _ = self.decoder2(decoded)
        decoded, _ = self.decoder3(decoded)
        return decoded
