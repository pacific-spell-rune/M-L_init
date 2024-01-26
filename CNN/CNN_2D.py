class Convolution2D:

    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        # weight size: (F, C, K, K)
        # bias size: (F) 
        self.F = num_filters
        self.K = kernel_size
        self.C = inputs_channel

        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0,self.F):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.C*self.K*self.K)), size=(self.C, self.K, self.K))

        self.p = padding
        self.s = stride
        self.lr = learning_rate
        self.name = name

    def zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = 2 * size + w
        new_h = 2 * size + h
        out = np.zeros((new_w, new_h))
        out[size:w+size, size:h+size] = inputs
        return out

    def forward(self, inputs):
        # input size: (C, W, H)
        # output size: (N, F ,WW, HH)
        C = inputs.shape[0]
        W = inputs.shape[1]+2*self.p
        H = inputs.shape[2]+2*self.p
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.p)
        WW = (W - self.K)/self.s + 1
        HH = (H - self.K)/self.s + 1
        feature_maps = np.zeros((self.F, WW, HH))
        for f in range(self.F):
            for w in range(0, WW, self.s):
                for h in range(0, HH, self.s):
                    feature_maps[f,w,h]=np.sum(self.inputs[:,w:w+self.K,h:h+self.K]*self.weights[f,:,:,:])+self.bias[f]

        return feature_maps

    def backward(self, dy):

        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(0, W, self.s):
                for h in range(0, H, self.s):
                    dw[f,:,:,:]+=dy[f,w,h]*self.inputs[:,w:w+self.K,h:h+self.K]
                    dx[:,w:w+self.K,h:h+self.K]+=dy[f,w,h]*self.weights[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
