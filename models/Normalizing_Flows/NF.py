class Normflow(nn.Module):
  def __init__(self, n_cdf: int):
    super(Normflow, self).__init__()

    self.mus=nn.Parameter(torch.rand(n_cdf), requires_grad=True)
    self.log_sigmas=nn.Parameter(torch.zeros(n_cdf), requires_grad=True)
    self.weights=nn.Parameter(torch.ones(n_cdf), requires_grad=True)

  def forward(self, x):
    distribution=Normal(self.mus, self.log_sigmas.exp())
    weights=torch.sigmoid(self.weights)
    z=(distribution.cdf(x)*weights).sum()
    dz=(distribution.log_prob(x)*weights).sum()
    return z, dz


def loss(distribution, z, dz):
  log_x=distribution.log_prob(z)+distribution.log_prob(dz)
  return log_x
