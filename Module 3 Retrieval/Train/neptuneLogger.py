import neptune.new as neptune
import time

class NetpuneLogger:

    def __init__(self, hyper):
        self.run = neptune.init(project='apalese/TrainingNNRetrieval',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1YmE0NjU5Ni1lZDk4LTRkMmEtYjIwYS02YjM5NGJmYWNlMmQifQ==',
                           source_files='config.json')

        self.run["JIRA"] = "NPT-952"
        self.run["algorithm"] = "TripletLossTrain"
        seconds = time.time()
        local = time.ctime(seconds)
        self.logname = 'log_' + local +'.txt'

        params = hyper
        self.run["parameters"] = params

    def set_train(self, l):
        self.run["train/loss"].log(l)

    def set_validation(self, l):
        self.run["train/validation_loss"].log(l)

    def update(self, weights_path, category):
        self.run["model-weights-" + category].upload(weights_path)

    def logfile(self, loss):
        seconds = time.time()
        local = time.ctime(seconds)
        with open(self.logname, 'a') as log:
            log.write(f"Registered Log : {local}\n \tweights modified due to loss : {loss}\n")
        self.run["logfile"].upload(self.logname)

    def destroy(self):
        self.run.stop()
