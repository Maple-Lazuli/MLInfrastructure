from dataclasses import dataclass
import requests
import threading

from src.app import start
from src.trainer import Trainer
from src.evaluator import Evaluator


@dataclass
class Manager:
    models: any = None
    data_manager: any = None
    ip: str = "0.0.0.0"
    port: int = 5000
    epochs: int = 1
    start_watcher_app: bool = True

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
            trainer.train()
            evaluator.evaluate()


if __name__ == "__main__":
    manager = Manager()

    manager.start_watcher()
