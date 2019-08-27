from flask import Flask, render_template
from flask import jsonify
from flask import request

from SentimentNet import SentimentAnalyzer
SN = SentimentAnalyzer(cuda=False)
app = Flask(__name__)

@app.route("/sentiment", methods=['GET','POST'])
def sentiment():
    if request.method=="GET":
        sent = request.args.get('sentence')
        print(sent)
        s = SN([sent])
        return str(s.item())
    if request.method=="POST":
        sent = request.form['sentence']
        print("POST "+sent)
        s = SN([sent])
        return str(sent)+":"+str(s.item())

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=9090,debug=True)
