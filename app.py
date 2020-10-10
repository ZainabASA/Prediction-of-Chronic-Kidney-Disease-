from flask import Flask,request,Response,render_template
import numpy as np
import joblib 
  

app = Flask(__name__,static_folder='static',template_folder='templates')			
modelxgb = joblib.load('FinalProject_model_XGB.pkl')

@app.route("/")
def index():
	return render_template("index.html")

@app.route('/Classify',methods=["POST"])
def Classify():
	specific_gravity = float (request.form.get("specific_gravity"))
	Suger = float (request.form.get("Suger"))						
	Pus_Cell = float (request.form.get("Pus_Cell"))					  
	Pus_Cell_clumps = float (request.form.get("Pus_Cell_clumps"))
	Bacteria = float (request.form.get("Bacteria"))				  	
	Blood_Glucose_Random = float (request.form['Blood_Glucose_Random'])
	Hemoglobin = float (request.form['Hemoglobin'])
	Packed_Cell_Volume = float (request.form['Packed_Cell_Volume'])
	White_Blood_Cell_Count = float (request.form['White_Blood_Cell_Count'])
	Red_Blood_Cell_Count = float (request.form['Red_Blood_Cell_Count'])
	Hypertension = float (request.form.get("Hypertension"))
	Diabetes_Mellitus = float (request.form.get("Diabetes_Mellitus"))					  	
	Coronary_Artery = float (request.form.get("Coronary_Artery"))					  	
	Appetite = float (request.form.get("Appetite"))					  	
	fv=[specific_gravity, Suger, Pus_Cell,Pus_Cell_clumps,Bacteria, Blood_Glucose_Random, Hemoglobin,Packed_Cell_Volume,White_Blood_Cell_Count,Red_Blood_Cell_Count, Hypertension,Diabetes_Mellitus,Coronary_Artery,Appetite]
	fv = np.array(fv).reshape((1,-1))
	res= str(modelxgb.predict(fv)[0])
	#res= int(round(modelr.predict([[Quality, Material, basement, totalBasementArea,FirstfloorName,livingAreaName,bathroomsName,Kitchen, GarageAreaName,GarageCarsName]])[0]))
	return render_template("index.html", Classify=res)
if __name__ == "__main__":
	app.run()
