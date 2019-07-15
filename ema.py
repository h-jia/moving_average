class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        
    def forward(self,x, last_average):
        new_average = self.mu*x + (1-self.mu)*last_average
        return new_average

ema = EMA(0.999)
x = Variable(torch.rand(5),requires_grad=True)
average = Variable(torch.zeros(5),requires_grad=True)
average = ema(x, average)


#############################################################

 class EMA(nn.Module):
     def __init__(self, mu):
         super(EMA, self).__init__()
         self.mu = mu
         self.shadow = {}

     def register(self, name, val):
         self.shadow[name] = val.clone()

     def forward(self, name, x):
         assert name in self.shadow
         new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
         self.shadow[name] = new_average.clone()
         return new_average

 ema = EMA(0.999)
 for name, param in model.named_parameters():
     if param.requires_grad:
         ema.register(name, param.data)

# in batch training loop
# for batch in batches:
     optimizer.step()
     for name, param in model.named_parameters():
         if param.requires_grad:
              param.data = ema(name, param.data)
