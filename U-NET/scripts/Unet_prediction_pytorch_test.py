import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from rasterio.transform import Affine

selected_gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
# Set random seed to ensure same results over each run
tf.random.set_seed(28)

# Paths
allimg = "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/2_Unet_on_floraMask/Mel_project/Data/" #CHANGE THE DIRECTORY TO BE UPDATED WITH LARGER TILES

os.chdir("/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/2_Unet_on_floraMask/Mel_project/pred/")

# List images
allimages = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(allimg) for f in filenames if f.endswith('.tif') and 'prediction' not in f])
# List shapes
#allshapes = sorted([os.path.join(dp, f) for dp, dn, filenames in os.walk(allimg_shap) for f in filenames if f.endswith('Copy.shp')])

# Parameters
res = 256
no_bands = 3
classes = 6
partshape = 1

# Load the best model
#def load_best_model(path):
#    model_files = sorted([f for f in os.listdir(path) if f.endswith('.hdf5')], key=lambda x: float(x.split('-')[2]))
#    best_model_file = model_files[0]
#    print(f"Loaded model from {best_model_file}.")
#    return tf.keras.models.load_model(os.path.join(path, best_model_file), compile=False)

# Load model
#model_path = os.getcwd()
#model = load_best_model(model_path)
model_path = "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/2_Unet_on_floraMask/Mel_project/checkpoints/"
model = tf.keras.models.load_model(model_path)

# Select the moving window steps
#factor1 = 25 #10 pixel steps

print(f"Number of images: {len(allimages)}")
#print(f"Number of shapefiles: {len(allshapes)}")

# Prediction loop
for t, img_path in enumerate(tqdm(allimages, desc='Processing Images')):
    # Load the image
    with rasterio.open(img_path) as ortho:
        ortho_data = ortho.read([1, 2, 3])
        ortho_data = np.moveaxis(ortho_data, 0, -1) / 255.0  # Normalize to [0, 1]
    print("Processing orthoimage ", t)

    # Load reference data
#     shape_path = allshapes[t]
#     with fiona.open(shape_path, 'r') as shapefile:
#         shapes = [feature['geometry'] for feature in shapefile if feature['properties']['id'] == partshape]
#     mask = geometry_mask(shapes, transform=ortho.transform, invert=True, out_shape=(ortho.height, ortho.width))
#     ortho_data = ortho_data * mask[..., None]

    # Set the moving window steps
    step=round(res / 10)
    ind_col = np.arange(1, ortho.width, step)
    ind_row = np.arange(1, ortho.height, step)
    ind_grid = np.array(np.meshgrid(ind_col, ind_row)).T.reshape(-1, 2)

    # Create a matrix to store the predictions
    preds_matrix = np.zeros((len(ind_grid), 3), dtype=np.float32)

    # Prediction loop over one orthoimage
    for i, (col, row) in enumerate(ind_grid):
        print("looping over part ", i, " of orthoimage ", t)
        col_start, col_end = col, min(col + res, ortho.width)
        row_start, row_end = row, min(row + res, ortho.height)
        #if col_end <= ortho.width and row_end <= ortho.height:
        ortho_crop = ortho_data[row_start:row_end, col_start:col_end]
        #if ortho_crop.shape == (res, res, no_bands): #and what do we do about parts of the orthoimage that fail this condition? 
        if ortho_crop.size == 0:
            continue
        
        tensor_pic = tf.convert_to_tensor(ortho_crop, dtype=tf.float32)
        #tensor_pic = tf.image.resize(tensor_pic, (res, res)) # redundant bc of the if statement - may be necessary if we removed if statement
        tensor_pic = tf.image.resize_with_pad(tensor_pic, target_height=res, target_width=res)
        tensor_pic = tf.expand_dims(tensor_pic, axis=0)
        preds = model.predict(tensor_pic)
        preds_class = np.argmax(preds, axis=-1).squeeze()
        preds_matrix[i, :] = [(col_start + col_end) / 2, (row_start + row_end) / 2, preds_class]
    print("Loop ", i, "completed")

    # Prediction rasterizing
    transform = ortho.transform
    out_meta = ortho.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": ortho.height,
        "width": ortho.width,
        "transform": transform,
        "count": 1,
        "dtype": 'uint8',
        "crs": ortho.crs
    })

    with rasterio.open(f"cnn_prediction_{partshape}_{os.path.basename(img_path)}", 'w', **out_meta) as dst:
        dst.write(preds_matrix[:, 2].reshape((len(ind_row), len(ind_col))).astype(rasterio.uint8), 1)
