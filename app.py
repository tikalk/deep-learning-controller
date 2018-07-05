import os
from flask import Flask, render_template, url_for, send_from_directory
from flask import jsonify
from flask import request
from flask_cors import CORS
from flask_uploads import UploadSet, configure_uploads, IMAGES
from boto import create_file_in_s3, create_file_in_local
import numpy as np

app = Flask(__name__)
app.config.from_pyfile('config.py', silent=True)
app.config['UPLOADED_PHOTOS_DEST'] = 'images'
photos = UploadSet('photos', IMAGES)

configure_uploads(app, photos)

CORS(app, automatic_options=True)


@app.route('/')
def space_invaders():
    return render_template('%s.html' % "spaceinvaders")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/thanks')
def thanks():
    return "thanks"


@app.route('/upload', methods=['POST'], )
def save_photos():
    if app.config['SAVE_IMAGES']:
        for file in request.files.keys():
            if app.config['SAVE_TARGET'] == 'local':
                print("Saving images to local...")
                create_file_in_local(file, "a", request.files[file])
            elif app.config['SAVE_TARGET'] == 's3':
                print("Saving images to s3...")
                create_file_in_s3(file, "a", request.files[file])
            else:
                print("Target configuration invalid (%s). Skipping data saving." % app.config['SAVE_TARGET'])
    else:
        print("Skipping images saving.")

    return jsonify({"path": url_for('thanks')})


@app.route('/upload_mobilenet_pred', methods=['POST'])
def save_mobilenet_predictions():
    if request.json is None:
        print("Non JSON predictions. Quitting")
        return jsonify({"path": url_for('thanks')})

    if not app.config['SAVE_TENSORS']:
        print("Skipping tensor saving")
        return jsonify({"path": url_for('thanks')})

    body = request.json
    xs = body['xs']
    ys = body['ys']
    xs_rows, xs_columns = get_2d_shape(xs['shape'])
    ys_rows, ys_columns = get_2d_shape(ys['shape'])
    if xs_rows != ys_rows:
        print("X rows count (%d) are not equal to Y rows (%d)" % (xs_rows, ys_rows))
        jsonify({"error": "X rows count (%d) are not equal to Y rows (%d)" % (xs_rows, ys_rows)})

    print("Saving tensors of %d predictions" % xs_rows)

    # Save X
    xs_2d = np.reshape(xs['data'], (xs_rows, xs_columns))
    xs_data_file = open('data/xs.csv','ab')
    np.savetxt(xs_data_file, xs_2d, delimiter=",")
    xs_data_file.close()

    # Save Y
    ys_2d = np.reshape(ys['data'], (ys_rows, ys_columns))
    ys_data_file = open('data/ys.csv','ab')
    np.savetxt(ys_data_file, ys_2d, delimiter=",")
    ys_data_file.close()
    return jsonify({"path": url_for('thanks')})


def get_2d_shape(shape):
    rows = shape[0]
    columns = 1
    for i, num in enumerate(shape):
        if i == 0:
            continue
        columns *= num
    return rows, columns


if __name__ == '__main__':
    app.run(host="0.0.0.0")
