import sys
import numpy as np
from keras.models import load_model
from fileIO import load_imgs_for_prediction, save_report

if __name__ == '__main__':
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        sys.exit(1)

    data_path = f'./data/{job_id}'
    model_path = f'./model/model7'
    save_path = f'./save/{job_id}'
    batch_size = 8

    imgs, paths = load_imgs_for_prediction(data_path)
    Rad51_model = load_model(model_path, compile=True)
    y_pred = np.argmax((Rad51_model.predict(imgs / 255.)), axis=1)
    save_report(y_pred, paths, savepath=save_path)
