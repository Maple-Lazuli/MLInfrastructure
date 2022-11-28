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


@app.route('/')
@app.route('/loss')
def loss():
    return render_template('loss.html')


@app.route('/evaluation')
def evaluation():
    return render_template('eval.html')


@app.route('/controller')
def control():
    return render_template('control.html')

@app.route('/console')
def console():
    return render_template('console.html')


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
    global performance
    temp = tempfile.NamedTemporaryFile()
    with open(temp.name, 'w') as file_out:
        json.dump(performance, file_out)
    return send_file(temp.name)


@app.route('/shutdown', methods=['GET'])
def shutdown():
    os._exit(0)


@app.context_processor
def inject_load():
    return {'trainingLossJSON': get_loss_graph('training'), 'validationLossJSON': get_loss_graph('validation'),
            'trainingEvalJSON': get_eval_table('training'), 'validationEvalJSON': get_eval_table('validation')}


def get_loss_graph(mode):
    global performance
    loss_graph = {
        'data': [Scatter(x=performance[key][mode]['index'],
                         y=performance[key][mode]['values'],
                         name=key) for key in performance.keys()],
        'layout': {
            'title': f'<b> {mode.capitalize()} Loss </b>',
            'yaxis': {
                'title': "<b> Loss % </b>"
            },
            'xaxis': {
                'title': "<b> Epoch </b>"
            }
        }
    }
    return json.dumps(loss_graph, cls=plotly.utils.PlotlyJSONEncoder)


def get_eval_table(mode):
    global performance
    keys = ['Name', 'Accuracy', 'Classification Error', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'TP', 'FP',
            'TN', 'FN']
    table = Figure([Table(
        header=dict(
            values=keys,
            font=dict(size=12),
            align="left"
        ),
        cells=dict(
            values=get_evaluation_metrics(performance, mode),
            align="left")
    )
    ])
    return json.dumps(table, cls=plotly.utils.PlotlyJSONEncoder)


def get_evaluation_metrics(performance, mode):
    return [

        [key for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.mean(performance[key]['evaluation'][mode]['Accuracy']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.mean(performance[key]['evaluation'][mode]['Classification Error']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.mean(performance[key]['evaluation'][mode]['Precision']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.mean(performance[key]['evaluation'][mode]['Recall']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.mean(performance[key]['evaluation'][mode]['Specificity']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.mean(performance[key]['evaluation'][mode]['F1-Score']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.sum(performance[key]['evaluation'][mode]['TP']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.sum(performance[key]['evaluation'][mode]['FP']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.sum(performance[key]['evaluation'][mode]['TN']) for key in performance.keys() if
         len(performance[key]['evaluation'][mode].keys()) > 0],

        [np.sum(performance[key]['evaluation'][mode]['FN']) for key in performance.keys() if
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
