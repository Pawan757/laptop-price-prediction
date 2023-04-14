from flask import Flask ,render_template,request,redirect,url_for,jsonify
import numpy as np
import json
import pickle


with open('artifacts\project_data.json','r') as file:
    project_data = json.load(file)

with open('artifacts\scale.pkl','rb') as file:
    scale = pickle.load(file)
    

with open('artifacts\model.pkl','rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/get_data',methods =['post'])
def get_data():
    data=request.form
    NAME= data['html_name']
    OS=data['html_os']
    PROCESSOR=data['html_processor']
    RAM =data['html_ram']
    STORAGE=data['html_storage']
    DISPLAY_INCH=data['html_display_inch']

    user_data = np.zeros(len(project_data['column_names']))

    search_name = 'name_'+NAME
    index = np.where(np.array(project_data['column_names']) == search_name)
    user_data[index] = 1

    search_os = 'os_'+OS
    index = np.where(np.array(project_data['column_names']) == search_os)
    user_data[index] = 1

    search_processor = 'processor_'+PROCESSOR
    index = np.where(np.array(project_data['column_names']) == search_processor)
    user_data[index] = 1

    search_ram = 'ram_'+RAM
    index = np.where(np.array(project_data['column_names']) == search_ram)
    user_data[index] = 1

    # user_data[0]= project_data['NAME'][NAME]
    # user_data[1]= project_data['OS'][OS]
    # user_data[2]= project_data['PROCESSOR'][PROCESSOR]
    # user_data[3]= project_data['RAM'][RAM]

    user_data[4]= STORAGE
    user_data[5]= DISPLAY_INCH


    user_data_scale = scale.transform([user_data])
    print(user_data_scale)
    
    result = model.predict(user_data_scale)[0]
    print(result)
    return jsonify({'prediction' : result})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
