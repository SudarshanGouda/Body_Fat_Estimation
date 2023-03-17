from flask import Flask,request,render_template
import sklearn

import pickle
import warnings
warnings.filterwarnings('ignore')

file1 = open('finalized_model.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        my_dict = request.form

        Density = float(my_dict['Density'])
        Weight = float(my_dict['Weight'])
        Height = float(my_dict['Height'])
        Chest = float(my_dict['Chest'])
        Abdomen = float(my_dict['Abdomen'])
        Thigh = float(my_dict['Thigh'])
        Knee = float(my_dict['Knee'])

        input_features = [[Density, Weight, Height, Chest, Abdomen, Thigh, Knee]]
        prediction = rf.predict(input_features)[0].round(2)

        # <p class="big-font">Hello World !!</p>', unsafe_allow_html=True

        string = 'Percentage of Body Fat Estimated is : ' + str(prediction)+'%'

        return render_template('show.html', string=string)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)