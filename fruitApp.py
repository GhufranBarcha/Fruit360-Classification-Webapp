from flask import Flask ,render_template ,request ,url_for
import joblib
import tensorflow



app = Flask(__name__)

## import model and encoder 
model1   =  joblib.load("fruit360model.pkl")
encoder1 = joblib.load("LabelEncoder.pkl")

@app.route('/')
def main():
    return render_template("index.html")



@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        img = request.files["image"]
        print(img)  # Print the uploaded image object
        pass
    return render_template("index.html", predicted="predicted~")




    return  render_template("index.html" , predicted = "predicted~")


if __name__ == "__main__":
    app.run(port = 3000 ,debug =True)    