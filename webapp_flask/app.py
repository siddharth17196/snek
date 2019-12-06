from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import getpreds

upload_folder = 'static'
allowed_extensions = ['jpeg' ,'jpg', 'png']

app = Flask(__name__)
app.config['upload_folder'] = upload_folder
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

def allowed_filename(filename):
	if not "." in filename:
		return False
	ext = filename.rsplit(".", 1)[1]
	if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
		return True
	else:
		return False


@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_filename(file.filename):
			filename = secure_filename(file.filename)
			img_loc = os.path.join(app.config['upload_folder'], filename)
			file = request.files['file']
			file.save(upload_folder+'\\'+'snek.jpg')
			print(img_loc)
			# os.chdir(upload_folder)
			# os.rename(filename,'snek.jpg')
			species = getpreds.get_item(upload_folder+'\\'+'snek.jpg')
			print(species)
			return render_template('upload_image.html', label=species, imgloc='../static/'+'snek.jpg' )#, imagesource=upload_folder+'\\'+filename)   
	return render_template("upload_image.html", imgloc='../static/'+'snek.jpg')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     species = getpreds.get_item(upload_folder)
#     if species != None:
#     	return species
#     else:
#     	return 'YOOOOOOOOOOOO'	

if __name__ == "__main__":
	app.run(debug=True)