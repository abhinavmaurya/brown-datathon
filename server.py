from flask import Flask, render_template, request
from werkzeug import secure_filename
# from sklearn import SVC

app = Flask(__name__, template_folder='template', static_folder="public")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['files[]']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)