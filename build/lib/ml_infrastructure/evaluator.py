import datetime
import numpy as np
import torch
import requests
from dataclasses import dataclass


def calc_performance(matrix, model, classes, mode):
    performance = {
        "name": model.name,
        "time": str(datetime.datetime.now()),
        "mode": mode,
        "Accuracy": [],
        "Classification Error": [],
        "TP": [],
        "FP": [],
        "TN": [],
        "FN": [],
        "Precision": [],
        "Recall": [],
        "Specificity": [],
        "F1-Score": [],
        "confusion_matrix": matrix.tolist(),
        "Classes": classes
    }

    for idx in range(len(classes)):
        TP = int(matrix[idx, idx])
        FN = int(np.sum(matrix[:, idx]) - TP)
        FP = int(np.sum(matrix[idx, :]) - TP)
        TN = int(np.sum(matrix) - (FN + FP + TP))
        performance["TP"].append(TP)
        performance["FN"].append(FN)
        performance["FP"].append(FP)
        performance["TN"].append(TN)
        if (TP + FP) != 0:
            performance['Precision'].append(TP / (TP + FP))
        else:
            performance['Precision'].append(0)

        if (TP + FN) != 0:
            performance['Recall'].append(TP / (TP + FN))
        else:
            performance['Recall'].append(0)

        if (TN + FP) != 0:
            performance['Specificity'].append(TN / (TN + FP))
        else:
            performance['Specificity'].append(0)

        if (TP + FP + TN + FN) != 0:
            performance['Accuracy'].append((TP + TN) / (TP + FP + TN + FN))
            performance['Classification Error'].append((FP + FN) / (TP + FP + TN + FN))
        else:
            performance['Accuracy'].append(0)
            performance['Classification Error'].append(0)

        if (performance['Precision'][idx] + performance['Recall'][idx]) != 0:
            performance['F1-Score'].append(((2 * performance['Precision'][idx] * performance['Recall'][idx]) /
                                            (performance['Precision'][idx] + performance['Recall'][idx])))
        else:
            performance['F1-Score'].append(0)

    performance['Total Accuracy'] = ((np.sum(performance['TP']) + np.sum(performance['TN'])) /
                                     (np.sum(performance['TP'])
                                      + np.sum(performance['TN'])
                                      + np.sum(performance['FP'])
                                      + np.sum(performance['FN'])))

    return performance


def send_performance(performance_dict, ip, port):
    server = f'http://{ip}:{port}/'
    endpoint = "/evalUpdate"

    body = {
        'data': performance_dict
    }

    url = server + endpoint
    requests.post(url, json=body)


def evaluate_model(model, data_manager, mode):
    classes = data_manager.classes

    if mode == 'training':
        loader = data_manager.train_loader
    else:
        loader = data_manager.validation_loader

    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    with torch.no_grad():
        for data in loader:
            inputs, targets = data[0], data[1]

            outputs = model.classify(inputs)

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to("cpu")

            for pred, true in zip(predicted, targets):
                confusion_matrix[int(pred), int(true)] += 1

    return confusion_matrix


@dataclass
class Evaluator:
    model: any = None
    data_manager: any = None
    ip: str = "0.0.0.0"
    port: int = 5000
    best_f1: int = 0

    def evaluate(self):
        total_performance = {'training': None, 'validation': None}
        for mode in ['training', 'validation']:
            matrix = evaluate_model(self.model, self.data_manager, mode)
            performance = calc_performance(matrix, self.model, self.data_manager.classes, mode)
            send_performance(performance, self.ip, self.port)

            total_performance[mode] = performance

        self.model.save(mode="current")

        if np.mean(total_performance['validation']['F1-Score']) > self.best_f1:
            self.model.save(mode='best')
            self.best_f1 = np.mean(total_performance['validation']['F1-Score'])
