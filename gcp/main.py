from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

Bucket_NAME = "danielputra-tfmodels"
class_names = ["Early Blight", "Late Blight", "Healthy", "Leaf Mold"]

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Model downloaded to {destination_file_name}")
    except Exception as e:
        print(f"Error downloading blob: {str(e)}")
        raise

def predict(request):
    global model
    if model is None:
        try:
            download_blob(
                Bucket_NAME,
                "tomato-imagegen.h5",
                "/tmp/tomato.h5",
            )
            model = tf.keras.models.load_model("/tmp/tomato.h5")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return {"error": "Failed to load model."}, 500

    try:
        image_file = request.files["file"]
        image = Image.open(image_file).convert("RGB").resize((256, 256))
        image = np.array(image) / 255.0
        image_array = tf.expand_dims(image, 0)

        predictions = model.predict(image_array)
        print(predictions)

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": "Failed to process prediction."}, 500
