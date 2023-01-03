import os
from dataclasses import dataclass
from http.client import RemoteDisconnected

import numpy as np
import requests
import threading
from collections import deque

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
    window_size :int = 20

    def __post_init__(self):
        self.trainers = [Trainer(model, self.data_manager, self.ip, self.port, self.epochs) for model in self.models]
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
            while True:
                loss = trainer.train()
                evaluator.evaluate()

                validation_loss = deque(maxlen=self.window_size)

                [validation_loss.append(l) for l in loss['validation_loss']]
                if len(validation_loss) == self.window_size:
                    slope = np.polyfit(range(0, len(validation_loss)), validation_loss, 1)[0]
                    print(slope)
                    if -0.01 < slope:
                        trainer.model.save(mode="Final")
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
            file_out.writelines(text)


if __name__ == "__main__":
    manager = Manager()

    manager.start_watcher()