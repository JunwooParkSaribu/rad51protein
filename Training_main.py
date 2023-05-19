import sys
import time
import matplotlib.pyplot as plt
import modelSave
from imageGenerator import ImgGenerator
from fileIO import imgs_to_ndarray, data_recur_search, subsampling
from model import ConvModel, Callback


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cur_path = sys.argv[1]
    else:
        cur_path = '.'

    data_path = '/shared/projects/histoneclf/rad51data/augmented_data'
    model_path = f'{cur_path}/model'
    epochs = 500
    batch_size = 8

    gpus = ConvModel.tf.config.list_physical_devices('GPU')
    print('GPUS LIST : ',gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                ConvModel.tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = ConvModel.tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print(f'\nLoading the data...')
    data_list = data_recur_search(data_path)
    images, labels = imgs_to_ndarray(data_list)
    images, labels = subsampling(images, labels, size=30000)

    print(images.shape, labels.shape)
    print(f'Generator building...')
    gen = ImgGenerator.DataGenerator(images, labels, ratio=0.8)
    train_ds = ConvModel.tf.data.Dataset.from_generator(gen.train_generator,
                                                        output_signature=(
                                                            ConvModel.tf.TensorSpec(
                                                                shape=(gen.get_shape()),
                                                                dtype=ConvModel.tf.float32),
                                                            ConvModel.tf.TensorSpec(
                                                                shape=(),
                                                                dtype=ConvModel.tf.int8))
                                                        ).batch(batch_size, drop_remainder=False)
    test_ds = ConvModel.tf.data.Dataset.from_generator(gen.test_generator,
                                                       output_signature=(
                                                           ConvModel.tf.TensorSpec(
                                                               shape=(gen.get_shape()),
                                                               dtype=ConvModel.tf.float32),
                                                           ConvModel.tf.TensorSpec(
                                                               shape=(),
                                                               dtype=ConvModel.tf.int8))
                                                       ).batch(batch_size, drop_remainder=False)

    print(f'Training the data...')
    training_model = ConvModel.Rad51(end_neurons=3)
    training_model.build(input_shape=(None, gen.get_shape()[0], gen.get_shape()[1], gen.get_shape()[2]))
    training_model.summary()
    training_model.compile(optimizer=ConvModel.tf.keras.optimizers.Adam(learning_rate=1e-5))
    history = training_model.fit(train_ds, validation_data=test_ds, epochs=epochs,
                                 callbacks=[Callback.EarlyStoppingAtMinLoss(patience=30),
                                            Callback.LearningRateScheduler()],
                                 trace='test_loss')  # training_loss, training_test_loss, test_loss

    model_name = modelSave.write_model_info(training_model, model_path, history,
                                            f'{time.gmtime().tm_mday}/{time.gmtime().tm_mon}/{time.gmtime().tm_year}, '
                                            f'{time.gmtime().tm_hour + 1}:{time.gmtime().tm_min}')
    print(f'{model_name} saved...')

    # loss history figure save
    plt.figure()
    plt.plot(range(0, len(history[0])), history[0], label='Training loss')
    plt.plot(range(0, len(history[1])), history[1], label='Validation loss')
    plt.legend()
    plt.savefig(f'{model_path}/{model_name}/loss_history.png')

    plt.figure()
    plt.plot(range(0, len(history[2])), history[2], label='Training acc')
    plt.plot(range(0, len(history[3])), history[3], label='Validation acc')
    plt.legend()
    plt.savefig(f'{model_path}/{model_name}/acc_history.png')
