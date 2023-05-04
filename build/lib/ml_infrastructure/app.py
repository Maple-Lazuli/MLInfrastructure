import json
import plotly
import tempfile

from flask import Flask, render_template, request, Response, send_file
from turbo_flask import Turbo
from plotly.graph_objs import Figure, Table, Scatter
import numpy as np
import os

app = Flask(__name__)
turbo = Turbo(app)

performance = dict()
idx = 0
stop_criteria = {
    "threshold": -1,
    "epoch": -1
}


@app.route('/')
@app.route('/loss')
def loss():
    return render_template('loss.html')


@app.route('/evaluation')
def evaluation():
    global idx
    idx = int(request.args['index'])
    print(idx)
    return render_template('eval.html')


@app.route('/controller')
def control():
    return render_template('control.html')


@app.route('/console')
def console():
    return render_template('console.html')


@app.route('/stop')
def stopForm():
    return render_template('stopForm.html')


@app.route('/stopCriteria')
def getStopCriteria():
    global stop_criteria

    return Response(json.dumps(stop_criteria), status=200, mimetype='application/json')


@app.route('/updateCriteria')
def updateCriteria():
    global stop_criteria

    if request.args['index'] == "":
        index = 0
    else:
        index = request.args['index']

    if request.args['threshold'] == "":
        threshold = -1
    else:
        threshold = request.args['threshold']

    if request.args['epoch'] == "":
        epoch = -1
    else:
        epoch = request.args['epoch']

    stop_criteria = {
        'metric': request.args['metric'],
        'threshold': threshold,
        'index': index,
        'epoch': epoch
    }

    return Response("Okay", status=200, mimetype='application/json')


@app.route('/updateLoss', methods=["POST"])
def updateLoss():
    global performance
    name = request.json['data']['name']
    mode = request.json['data']['mode']
    if name not in performance.keys():
        performance = add_model(performance, name)

    performance[name][mode]['values'].append(request.json['data']['value'])
    performance[name][mode]['index'].append(request.json['data']['index'])

    if mode == "training":
        turbo.push(turbo.update(render_template('trainingLoss.html'), 'trainingLoss'))
    else:
        turbo.push(turbo.replace(render_template('validationLoss.html'), 'validationLoss'))

    return Response("Okay", status=200, mimetype='application/json')


@app.route('/evalUpdate', methods=["POST"])
def evalUpdate():
    global performance
    evaluation_results = request.json['data']
    name = evaluation_results['name']
    mode = evaluation_results['mode']

    if name not in performance.keys():
        performance = add_model(performance, name)
    performance[name]['evaluation'][mode] = evaluation_results

    if mode == 'training':
        turbo.push(turbo.replace(render_template('trainingEval.html'), 'trainingEval'))
    else:
        turbo.push(turbo.replace(render_template('validationEval.html'), 'validationEval'))

    return Response("Okay", status=200, mimetype='application/json')


@app.route('/save', methods=['GET'])
def save():
    global performance
    save_location = request.args.get('save_location')
    save_name = request.args.get('save_name')
    full_name = os.path.join(save_location, save_name)
    with open(full_name, 'w') as file_out:
        json.dump(performance, file_out)

    return Response(f"Saved: {full_name}", status=200, mimetype='application/json')


@app.route('/restore', methods=['POST'])
def restore():
    global performance
    performance = request.json['data']
    turbo.push(turbo.replace(render_template('trainingEval.html'), 'trainingEval'))
    turbo.push(turbo.replace(render_template('validationEval.html'), 'validationEval'))
    turbo.push(turbo.update(render_template('trainingLoss.html'), 'trainingLoss'))
    turbo.push(turbo.replace(render_template('validationLoss.html'), 'validationLoss'))
    return Response(f"Restored", status=200, mimetype='application/json')


@app.route('/registerModel', methods=['POST'])
def register_model():
    global performance
    model_data = request.json['data']
    name = model_data['name']
    if name not in performance.keys():
        performance = add_model(performance, name)
    performance[name]['model_details'] = model_data

    return Response(f"Registered {name}", status=200, mimetype='application/json')


@app.route('/reset', methods=['GET'])
def reset():
    global performance
    performance = dict()
    return Response(f"Reset Success!", status=200, mimetype='application/json')


@app.route('/download', methods=['GET'])
def download():
    # global performance
    # temp = tempfile.NamedTemporaryFile()
    # with open(temp.name, 'w') as file_out:
    #     json.dump(performance, file_out)
    # return send_file(temp.name, download_name="Evaluation_Metrics.json")
    global performance
    return Response(json.dumps(performance), status=200, mimetype='application/json')


@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


@app.context_processor
def inject_load():
    return {'trainingLossJSON': get_loss_graph('training'), 'validationLossJSON': get_loss_graph('validation'),
            'trainingLogLossJSON': get_loss_graph_log('training'),
            'validationLogLossJSON': get_loss_graph_log('validation'), 'trainingEvalJSON': get_eval_table('training'),
            'validationEvalJSON': get_eval_table('validation')}


def get_loss_graph(mode):
    global performance
    loss_graph = {
        'data': [Scatter(x=performance[key][mode]['index'],
                         y=performance[key][mode]['values'],
                         name=key) for key in performance.keys()],
        'layout': {
            'title': f'<b> {mode.capitalize()} Loss </b>',
            'yaxis': {
                'title': "<b> Loss </b>"
            },
            'xaxis': {
                'title': "<b> Epoch </b>"
            }
        }
    }
    return json.dumps(loss_graph, cls=plotly.utils.PlotlyJSONEncoder)


def get_loss_graph_log(mode):
    global performance
    loss_graph = {
        'data': [Scatter(x=performance[key][mode]['index'],
                         y=[np.log10(i) if i != 0 else np.log10(.1e-100) for i in performance[key][mode]['values']],
                         name=key) for key in performance.keys()],
        'layout': {
            'title': f'<b> {mode.capitalize()} Log Loss </b>',
            'yaxis': {
                'title': "<b> Log Loss </b>"
            },
            'xaxis': {
                'title': "<b> Epoch </b>"
            }
        }
    }
    return json.dumps(loss_graph, cls=plotly.utils.PlotlyJSONEncoder)


def get_eval_table(mode):
    global performance
    global idx
    keys = ['Name', 'Accuracy', 'Classification Error', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'TP', 'FP',
            'TN', 'FN']
    table = Figure([Table(
        header=dict(
            values=keys,
            font=dict(size=12),
            align="left"
        ),
        cells=dict(
            values=get_evaluation_metrics(performance, mode, idx),
            align="left")
    )
    ])
    return json.dumps(table, cls=plotly.utils.PlotlyJSONEncoder)


# TODO Verify this function works
def get_evaluation_metrics(performance, mode, idx=0):
    return [

        [key for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['Accuracy'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['Classification Error'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['Precision'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['Recall'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['Specificity'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['F1-Score'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['TP'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['FP'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['TN'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [performance[key]['evaluation'][mode]['FN'][idx] for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],
    ]


def add_model(performance_dict, name):
    performance_dict[name] = {'training': {'values': [], 'index': []},
                              'validation': {'values': [], 'index': []},
                              'evaluation': {'training': dict(), 'validation': dict()},
                              'model_details': dict()}
    return performance_dict


def start(ip='0.0.0.0', port=5000):
    app.run(host=ip, port=port)


if __name__ == "__main__":
    start()
