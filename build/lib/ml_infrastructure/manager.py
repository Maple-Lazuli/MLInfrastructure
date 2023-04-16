from collections import deque
from dataclasses import dataclass
from http.client import RemoteDisconnected
import json
import os
import requests
import threading

import numpy as np

from ml_infrastructure.app import start
from ml_infrastructure.trainer import Trainer
from ml_infrastructure.evaluator import Evaluator


@dataclass
class Manager:
    models: any = None
    data_manager: any = None
    ip: str = "0.0.0.0"
    port: int = 5000
    epochs: int = 100
    start_watcher_app: bool = True
    window_size: int = 20
    eval_rate: int = 1

    def __post_init__(self):
        if self.models is not None:
            self.trainers = [Trainer(model, self.data_manager, self.ip, self.port, self.epochs) for model in
                             self.models]
            self.evaluators = [Evaluator(model, self.data_manager, self.ip, self.port) for model in self.models]

        if self.start_watcher_app:
            self.start_watcher()

    def start_watcher(self):
        threading.Thread(target=start, args=(self.ip, int(self.port),)).start()

    def register_model(self, model):
        post_body = {
            'data': model.get_model_info()
        }
        url = f'http://{self.ip}:{self.port}/registerModel'

        requests.post(url=url, json=post_body)

    def perform(self):

        for trainer, evaluator in zip(self.trainers, self.evaluators):
            epoch_num = 0
            while True:
                epoch_num += 1
                loss = trainer.train()

                if min(loss['validation_loss']) == loss['validation_loss'][-1]:
                    trainer.model.save(mode="Lowest-Val-Loss", ip=self.ip, port=self.port)

                if epoch_num%self.eval_rate != 0:
                    continue

                performance = evaluator.evaluate()
                stopping_criteria = requests.get(f"http://{self.ip}:{self.port}/stopCriteria")
                stopping_criteria = json.loads(stopping_criteria.text)

                if stopping_criteria['threshold'] != -1 or stopping_criteria['epoch'] != -1:
                    if stopping_criteria['threshold'] != -1:
                        performance = performance['validation'][stopping_criteria['metric']][stopping_criteria['index']]
                        if performance >= stopping_criteria['threshold']:
                            trainer.model.save(mode="Final", ip=self.ip, port=self.port)
                            break
                    elif stopping_criteria['epoch'] != -1:
                        if epoch_num >= int(stopping_criteria['epoch']):
                            trainer.model.save(mode="Final", ip=self.ip, port=self.port)
                            break
                else:
                    validation_loss = deque(maxlen=self.window_size)
                    [validation_loss.append(l) for l in loss['validation_loss']]
                    if len(validation_loss) == self.window_size:
                        slope = np.polyfit(range(0, len(validation_loss)), validation_loss, 1)[0]
                        if -0.001 < slope:
                            trainer.model.save(mode="Final", ip=self.ip, port=self.port)
                            break

    def shutdown_watcher(self):
        try:
            requests.get(f'http://{self.ip}:{self.port}/shutdown')
        except RemoteDisconnected as e:
            print(f"Shutdown Watcher at {self.ip}:{self.port}")

    def get_watcher_results(self):
        res = requests.get(f'http://{self.ip}:{self.port}/download')
        return res.text

    def save_watcher_results(self, save_location, save_name):
        text = self.get_watcher_results()
        if not os.path.isdir(save_location):
            os.mkdir(save_location)

        with open(os.path.join(save_location, save_name), 'w') as file_out:
            json.dump(text, file_out)


if __name__ == "__main__":
    manager = Manager()

    manager.start_watcher()
