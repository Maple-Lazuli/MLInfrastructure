import json
import plotly

from flask import Flask
from flask import render_template, request, Response
from turbo_flask import Turbo
from plotly.graph_objs import Bar, Figure, Table, Scatter
import numpy as np

app = Flask(__name__)
turbo = Turbo(app)

# loss = {
#     'training': dict(),
#     'validation': dict()
# }

performance = dict()

session_details = dict()


@app.route('/')
@app.route('/loss')
def loss():
    return render_template('loss.html')

@app.route('/evaluation')
def evaluation():
    return render_template('eval.html')


@app.route('/updateLoss', methods=["POST"])
def updateLoss():
    global loss
    name = request.json['data']['name']
    mode = request.json['data']['mode']
    print(f'name: {name} mode: {mode}')
    if name in performance.keys():
        performance[name][mode]['values'].append(request.json['data']['value'])
        performance[name][mode]['index'].append(request.json['data']['index'])
    else:
        performance[name] = {'training': {'values': [], 'index': []},
                            'validation': {'values': [], 'index': []},
                            'evaluation': {'training': dict(), 'validation':dict()}}
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
    performance = request.json['data']

    if performance['mode'] == 'training':
        turbo.push(turbo.replace(render_template('trainingEval.html'), 'trainingEval'))
    else:
        turbo.push(turbo.replace(render_template('validationEval.html'), 'validationEval'))

    return Response("Okay", status=200, mimetype='application/json')



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
    keys = ['Name', 'Accuracy', 'Classification Error', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'TP', 'FP', 'TN', 'FN']
    table = Figure([Table(
        header=dict(
            values=keys,
            font=dict(size=12),
            align="left"
        ),
        cells=dict(
            values=[get_values(performance[model]['evaluation'][mode]) for model in performance.keys() if
                    performance[model]['evaluation'][mode] is not None],
            align="left")
    )
    ])

    return json.dumps(table, cls=plotly.utils.PlotlyJSONEncoder)

def get_values(performance_dict):
    return [
        performance_dict['name'],
        np.mean(performance_dict['Classification Error']),
        np.mean(performance_dict['Precision']),
        np.mean(performance_dict['Recall']),
        np.mean(performance_dict['Specificity']),
        np.mean(performance_dict['F1-Score']),
        np.sum(performance_dict['TP']),
        np.sum(performance_dict['FP']),
        np.sum(performance_dict['TN']),
        np.sum(performance_dict['FN']),
    ]

def start(ip='0.0.0.0', port=5000):
    app.run(host=ip, port=port)
