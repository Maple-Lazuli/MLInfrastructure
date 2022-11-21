import json
import plotly

from flask import Flask
from flask import render_template, request, Response
from turbo_flask import Turbo
from plotly.graph_objs import Bar, Figure, Table, Scatter

app = Flask(__name__)
turbo = Turbo(app)

loss = {
    'training': dict(),
    'validation': dict()
}
accuracy = {
    'training': dict(),
    'validation': dict()
}
confusion_matrix = dict()


@app.route('/')
def metrics():
    """
    """

    return render_template('metrics.html')


@app.route('/updateLoss', methods=["POST"])
def update():
    global loss
    name = request.json['data']['name']
    mode = request.json['data']['mode']
    print(f'name: {name} mode: {mode}')
    if name in loss[mode].keys():
        loss[mode][name]['values'].append(request.json['data']['value'])
        loss[mode][name]['index'].append(request.json['data']['index'])
    else:
        loss[mode][name] = dict()
        loss[mode][name]['values'] = [request.json['data']['value']]
        loss[mode][name]['index'] = [request.json['data']['index']]
    if mode == "training":
        turbo.push(turbo.update(render_template('trainingLoss.html'), 'trainingLoss'))
    else:
        turbo.push(turbo.replace(render_template('validationLoss.html'), 'validationLoss'))

    return Response("Okay", status=200, mimetype='application/json')


@app.context_processor
def inject_load():
    return {'trainingLossJSON': get_loss_graph('training'), 'validationLossJSON': get_loss_graph('validation')}


def get_loss_graph(mode):
    global loss
    loss_graph = {
        'data': [Scatter(x=loss[mode][key]['index'],
                      y=loss[mode][key]['values'],
                    name=key) for key in loss[mode].keys()],
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


def get_accuracy_graph():
    pass


def confusion_matrix():
    pass

