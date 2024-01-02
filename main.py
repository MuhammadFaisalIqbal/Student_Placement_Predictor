from flask import Flask
from flask import request
from flask import jsonify
import pickle
import numpy as np
import sklearn
print(sklearn.__version__)

model = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

# gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p, salary
@app.route('/predict', methods=['POST'])
def predict():
    # Receiving Inputs from the users as follows:
    gender = request.form.get("gender")
    ssc_p = request.form.get('ssc_p')
    ssc_b = request.form.get("ssc_b")
    hsc_p = request.form.get("hsc_p")
    hsc_b = request.form.get("hsc_b")
    hsc_s = request.form.get("hsc_s")
    degree_p = request.form.get("degree_p")
    degree_t = request.form.get("degree_t")
    workex = request.form.get("workex")
    etest_p = request.form.get("etest_p")
    specialisation = request.form.get("specialisation")
    mba_p = request.form.get("mba_p")
    salary = request.form.get("salary")

    # result = {'gender': gender, 'ssc_p': ssc_p, 'ssc_b': ssc_b, 'hsc_p': hsc_p,
    #           'hsc_b': hsc_b, 'hsc_s': hsc_s, 'degree_p': degree_p,
    #           'degree_t': degree_t, 'workex': workex, 'etest_p': etest_p,
    #           'specialisation': specialisation, 'mba_p': mba_p, 'salary': salary}

    input_query = np.array([[gender, ssc_p, ssc_b, hsc_p,
                             hsc_b, hsc_s, degree_p, degree_t,
                             workex, etest_p, specialisation,
                             mba_p, salary]])

    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)