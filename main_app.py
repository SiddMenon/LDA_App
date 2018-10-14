import flask
app = flask.Flask(__name__)

@app.route("/")
def index():
    #do whatevr here...
    print("Hello World")
    return "Return gets printed instead"
