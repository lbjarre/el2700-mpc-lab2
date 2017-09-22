from flask import Flask, make_response
import yaml
import numpy as np
from vehiclemodel import CarModel

app = Flask(__name__)

@app.route('/')
def disp_page():
    with app.open_resource('view.html') as f:
        return f.read()

@app.route('/fetch')
def send_sim_data():
    with open('python-code/carmodels.yaml') as input_file:
        car_models = yaml.safe_load(input_file)

    car = CarModel(car_models['car1'])

    time = np.arange(0, 10, 0.1)
    inputs = np.random.normal(0, 10, (len(time), 2))
    outputs = np.zeros((len(time), 4))

    for i, t in enumerate(time):
        car.update_state(inputs[i, :])
        outputs[i, :] = car.z

    return make_response(str(outputs))

if __name__ == '__main__':
    app.run()
