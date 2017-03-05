from flask import Flask, render_template, request
from werkzeug import secure_filename
# from sklearn import SVC
from sklearn.externals import joblib
from classifierLive import BookingPredictor

app = Flask(__name__, template_folder='template', static_folder="public")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['files[]']
        secure_fn = secure_filename(f.filename)
        f.save(secure_fn)
        p_result = predict(secure_fn)
        print(p_result)
        return p_result
        # return 'file uploaded successfully'


def predict(filename):
    bp = BookingPredictor(filename)
    return bp.run_voting_classifier()

if __name__ == '__main__':
    app.run(debug=True)