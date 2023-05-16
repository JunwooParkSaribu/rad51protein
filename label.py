def labeling(file):
    try:
        hint = file.split('/')[-2].lower()
        if '1_stiff' in hint:
            return 0
        elif '2_bent' in hint:
            return 1
        elif '3_circles' in hint:
            return 2
        elif '4_1not' in hint:
            return 3
        elif '5_others' in hint:
            return 4
        else:
            raise Exception
    except:
        print('Err while labeling')
