from flask import Flask,redirect,url_for,render_template,request
import numpy as np
from keras.models import model_from_json
import cv2

app=Flask(__name__)

key = cv2.waitKey(1)


json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("facialemotionmodel.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0


"""@app.route('/',methods=['GET','POST'])
def home():
        return render_template('login.html')


@app.route('/logincheck',methods=['GET','POST'])
def logincheck():
    if request.method=='POST':
        username = request.form["username"]
        password = request.form["password"]

        if username == "emotion" and  password == "emotion":
            return render_template("prediction.html")
        else:
            return render_template("index.html")
    return render_template('index.html')



@app.route('/sign',methods=['GET','POST'])
def sign():
    return render_template('login.html')"""

@app.route('/',methods=['GET','POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/happy',methods=['GET','POST'])
def happy():
    return render_template('happy.html')

@app.route('/sad',methods=['GET','POST'])
def sad():
    return render_template('sad.html')

@app.route('/angry',methods=['GET','POST'])
def angry():
    return render_template('angry.html')

@app.route('/nutral',methods=['GET','POST'])
def nutral():
    return render_template('neutral.html')

@app.route("/open_cam")
def open_cam():
    webcam=cv2.VideoCapture(0)
    labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
    while True:
        i,im=webcam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(im,1.3,5)
        try: 
            for (p,q,r,s) in faces:
                image = gray[q:q+s,p:p+r]
                cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2) 
                image = cv2.resize(image,(48,48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                filename = prediction_label
                print("Predicted Output:", prediction_label)
            
                # cv2.putText(im,prediction_label)
                cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
                cv2.waitKey(27)
                return render_template("prediction.html" ,e = prediction_label)
        except cv2.error:
            pass

@app.route('/song',methods=['GET','POST'])
def song():
    if request.method=='POST':
        emotion = request.form["emotion"]
        if(emotion == "happy"):
            return render_template("happy.html" , em = emotion)
        elif(emotion == "sad"):
            return render_template("sad.html" , em = emotion)
        elif(emotion == "angry"):
            return render_template("angry.html" , em = emotion)
        elif(emotion == "neutral"):
            return render_template("neutral.html" , em = emotion)
        elif(emotion == "fear"):
            return render_template("fear.html" , em = emotion)
        elif(emotion == "surprise"):
            return render_template("surprise.html" , em = emotion)
        elif(emotion == "disgust"):
            return render_template("disgust.html" , em = emotion)
    return render_template("index.html")


if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)