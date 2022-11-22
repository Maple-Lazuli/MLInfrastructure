import json
import plotly

from flask import Flask
from flask import render_template, request, Response
from turbo_flask import Turbo
from plotly.graph_objs import Bar, Figure, Table, Scatter

app = Flask(__name__)
turbo = Turbo(app)

# loss = {
#     'training': dict(),
#     'validation': dict()
# }

peformance = dict()

session_details = dict()


@app.route('/')
def metrics():
    return render_template('metrics.html')


@app.route('/updateLoss', methods=["POST"])
def updateLoss():
    global loss
    name = request.json['data']['name']
    mode = request.json['data']['mode']
    print(f'name: {name} mode: {mode}')
    if name in peformance.keys():
        peformance[name][mode]['values'].append(request.json['data']['value'])
        peformance[name][mode]['index'].append(request.json['data']['index'])
    else:
        peformance[name] = {'training': {'values': [], 'index': []},
                            'validation': {'values': [], 'index': []},
                            'evaluation': None}
        peformance[name][mode]['values'].append(request.json['data']['value'])
        peformance[name][mode]['index'].append(request.json['data']['index'])

    if mode == "training":
        turbo.push(turbo.update(render_template('trainingLoss.html'), 'trainingLoss'))
    else:
        turbo.push(turbo.replace(render_template('validationLoss.html'), 'validationLoss'))

    return Response("Okay", status=200, mimetype='application/json')


@app.route('/evalUpdate', methods=["POST"])
def evalUpdate():
    global performance
    performance = request.json['data']
    print(performance)
    #turbo.push(turbo.replace(render_template('validationLoss.html'), 'validationLoss'))

    return Response("Okay", status=200, mimetype='application/json')



@app.context_processor
def inject_load():
    return {'trainingLossJSON': get_loss_graph('training'), 'validationLossJSON': get_loss_graph('validation')}


def get_loss_graph(mode):
    global peformance
    loss_graph = {
        'data': [Scatter(x=peformance[key][mode]['index'],
                         y=peformance[key][mode]['values'],
                         name=key) for key in peformance.keys()],
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


def start(ip='0.0.0.0', port=5000):
    app.run(host=ip, port=port)
