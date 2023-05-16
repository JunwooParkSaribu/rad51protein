import os


def write_model_info(training_model, path: str, history: list, date: str) -> str:
    new_model_num = 0
    try:
        if os.path.isdir(path):
            contents = os.listdir(path)
            for content in contents:
                if 'model' in content:
                    model_num = int(content.split('_')[0].split('model')[-1])
                    new_model_num = max(new_model_num, model_num)
            modelname = f'model{new_model_num + 1}'
        training_model.save(f'{path}/{modelname}')
    except:
        print('model directory creation err')
        raise Exception

    with open(f'{path}/{modelname}/log.txt', 'w') as info_file:
        info_file.write(f'date: {date}\n')
        info_file.write(f'training loss, validation loss, training accuracy, validation accuracy\n')
        for line_num in range(len(history[0])):
            info_file.write(f'{str(history[0][line_num])}\t{str(history[1][line_num])}\t'
                            f'{str(history[2][line_num])}\t{str(history[3][line_num])}\n')

    return modelname
