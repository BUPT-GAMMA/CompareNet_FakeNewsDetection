class Trainer:
    def __init__(self, params, utils):
        self.params = params
        self.utils = utils
        self.log_time = {}

    def train(self):
        print('-----------{}-------------'.format(self.params.config))
        training_time = self.utils.train(save_plots_as=self.params.config)
        self.log_time[self.params.config] = training_time
        print('-----------------------------------------')
