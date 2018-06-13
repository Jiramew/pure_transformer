from model.layer import Layer


class AdditionLayer(Layer):
    def __init__(self,
                 name,
                 input_list,
                 network):
        super(AdditionLayer, self).__init__(name=name)
        self.inputs = input_list
        self.outputs = None

        self.network = network

        self.back_error = None

    def forward(self,
                inputs):
        outputs = sum([layer.forward(inputs) for layer in self.inputs])
        self.outputs = outputs
        return self.outputs

    def backward(self):
        self.back_error = self.next_layer.back_error
