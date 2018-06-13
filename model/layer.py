class Layer(object):
    def __init__(self, name):
        self.name = name

        self.next_layer = None
        self.pre_layer = None

        self.layer_index = None

    def is_last_layer(self):
        return self.next_layer is None

    def set_input_layer(self, input_layer):
        self.pre_layer = input_layer

    def set_output_layer(self, output_layer):
        self.next_layer = output_layer

    def batch_update(self):
        pass
