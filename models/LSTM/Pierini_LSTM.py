class AutoencoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoencoderLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder2 = nn.LSTM(hidden_size, 8, batch_first=True)
        self.decoder = nn.LSTM(8, hidden_size, batch_first=True)
        self.decoder2 = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded, _ = self.encoder2(encoded)
        decoded, _ = self.decoder(encoded)
        decoded, _ = self.decoder2(decoded)
        return decoded
