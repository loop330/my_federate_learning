def __init__(self,conf,eval_dataset):
    self.conf = conf
    self.global_model = models.get_model(self.conf["model_name"])