from func.operations1d import *
from func.sinc import SincConv_fast, Conv_0
from func.p2sgrad import P2SActivationLayer


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN_half(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN_same(C_prev, C, 1, 1, 0, affine=False)

    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)
    self.pooling_layer = nn.MaxPool1d(2)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)

      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    out = torch.cat([states[i] for i in self._concat], dim=1)
    out = self.pooling_layer(out)
    return out

class Network(nn.Module):

  def __init__(self, C, layers, args, num_classes, genotype):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self.is_mask = args.is_mask

    if args.sinc_scale == 'conv':
      self.sinc = Conv_0(C, kernel_size=args.sinc_kernel,
                          is_mask=args.is_mask)
      print("****** Initialising Conv blocks as front end ******")
    else:
      self.sinc = SincConv_fast(C, kernel_size=args.sinc_kernel, 
                            freq_scale=args.sinc_scale,
                            is_mask=args.is_mask,
                            is_trainable=args.is_trainable)
      print("****** Initialising Sinc filters as front end ******")

    self.mp = nn.MaxPool1d(3)
    self.bn = nn.BatchNorm1d(C)
    self.lrelu = nn.LeakyReLU(negative_slope=0.3)

    self.stem = nn.Sequential(
      nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm1d(C),
      nn.LeakyReLU(negative_slope=0.3),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      if reduction:
        print('Reduce Cell')
      else:
        print('Normal Cell')
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    self.gru = nn.GRU(input_size = C_prev,
			hidden_size = args.gru_hsize,
			num_layers = args.gru_layers,
			batch_first = True)
    self.fc_gru = nn.Linear(args.gru_hsize, args.gru_hsize)
    self.l_layer = P2SActivationLayer(args.gru_hsize, out_dim=2)
    
    

  def forward(self, input):

    input = input.unsqueeze(1)
    s0 = self.sinc(input, self.training)
    s0 = self.mp(s0)
    s0 = self.bn(s0)
    s0 = self.lrelu(s0)
    s1 = self.stem(s0)


    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    v = s1
    v = v.permute(0, 2, 1)
    self.gru.flatten_parameters()
    v, _ = self.gru(v)
    v = v[:,-1,:]
    embeddings = self.fc_gru(v)

    if not self.training:
      return embeddings
    else:
      logits = self.l_layer(embeddings)
      return logits, embeddings
  
  def forward_classifier(self, embeddings):
    logits = self.l_layer(embeddings)
    return logits




