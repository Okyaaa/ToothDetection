from flask import Flask, render_template, request, redirect, url_for, flash
import torch.nn as nn
import torch
from PIL import Image
import os
import io
from torchvision import transforms
import shutil


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = './static/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('main.html')

def get_prediction(img_bytes):
    model = torch.hub.load('./yolov5-master/', 'custom', path='./yolov5-master/best.pt', source='local', force_reload=True,)
    model.eval()

    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
    print("hello im here")

    # Inference
    results = model(imgs)  # includes NMS
    results.crop(save=True)
    return results

@app.route('/detect', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        file = request.files.get('file')
        
        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir='static')

        return redirect(request.url)
    return render_template('main.html')



def get_prediction_cnn(image):
    model_cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=False)
    model_cnn.eval()

    num_classes = 4
    model_cnn .fc = nn.Linear(model_cnn.fc.in_features, num_classes)
    model_cnn .load_state_dict(torch.load('./yolov5-master//RestNext-2.pth'))
    model_cnn .eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_cnn = Image.open(image)
    image_tensor = transform(img_cnn).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model_cnn(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # Load the class labels
    class_labels = ['Impaksi', 'Karies', 'LesiPeriapikal', 'Resorbsi']  # Replace with your actual class labels
    
    predicted_label = predicted.item()

    return class_labels[predicted_label]

@app.route('/detect_disease', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':

        file = request.files['file']


        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image_cnn.jpg'))

        image = file

        global results_cnn

        results_cnn = get_prediction_cnn(image)

        return redirect(request.url)
    return render_template('main.html', value = results_cnn)

@app.route('/delete_dir', methods=['GET', 'POST'])
def delete_dir():
    dir_path_runs = './runs'
    dir_path_static = './static'
    shutil.rmtree(dir_path_runs)
    shutil.rmtree(dir_path_static)

    return render_template('main.html')





# @app.route('/data')
# def get_data():
#     data = []
#     for i in range(32):
#         item = {
#             'Tooth Number': f'Person {i+1}',
#             'age': 20 + i,
#             'city': 'Some City'
#         }
#         data.append(item)
    
#     # Using jsonify to create a JSON response with an array
#     response = jsonify(data)
#     return response



# @app.route('/dir_name')
# def get_directory_name():
#     data = []
#     folder_path = 'runs/detect/exp/crops'  # Replace with the actual folder path
#     directories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
#     for i in range(len(directories)):
#         image_path = folder_path + '/' + directories[i]
#         image_path = image_path + '/' + os.listdir(image_path)[0]

#         image_path2 = image_path = 'D:/KulyahAkademik/ProyekAkhir/WebProject/static/image0.jpg'  # Replace with the actual image path
#         send_file(image_path2, mimetype='image/jpeg')

#         item = {
#             'Tooth Number': directories[i],
#             'Image': send_file(image_path, mimetype='image/jpeg')
#         }
#         data.append(item)
    
#     # Using jsonify to create a JSON response with an array
#     response = jsonify(data)
#     return response

if __name__ == '__main__':
    app.run(debug=True, port=8080)
