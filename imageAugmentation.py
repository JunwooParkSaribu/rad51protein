from fileIO import imgs_to_ndarray, data_recur_search, subsampling
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import imageio.v3 as iio
import imutils
from sklearn.svm import SVC


def dechanneling(image):
    new_image = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
    for r in range(len(image)):
        for c in range(len(image[r])):
            new_image[r][c] = image[r][c][0].copy()
    return new_image


def gaussreg(image, extended_image, ext_size=150):
    image = dechanneling(image)
    y_true = [x for x in image.reshape(-1)]
    predicted_y = []
    x = []
    for i in range(0, ext_size):
        for j in range(0, ext_size):
            if 50 <= i < 100 and 50 <= j < 100:
                x.append([i,j])
            predicted_y.append([i,j])
    predicted_y = np.array(predicted_y)
    x = np.array(x)

    reg = SVC(C=10, kernel='rbf').fit(X=x, y=y_true)
    y_pred = reg.predict(predicted_y).reshape(ext_size, ext_size)
    for i in range(0, ext_size):
        for j in range(0, ext_size):
            if 50 <= i < 100 and 50 <= j < 100:
                continue
            else:
                if y_pred[i][j] < 0:
                    y_pred[i][j] = 0
                extended_image[i][j] = y_pred[i][j].copy()


def rotation(image, nb=12):
    rotated_images = []
    for i in range(1, nb):
        deg = i * 360//nb
        rotated_image = imutils.rotate(image, angle=deg)
        rotated_images.append(rotated_image)
    return np.array(rotated_images)


def rotation_shift(images, data_list, shift_nb=10, rotation_nb=24, savepath='.'):
    data_list = [fname.split('/')[-1] for fname in data_list]
    for fname, (img_num, image) in zip(data_list, enumerate(images)):
        shifted_images = []
        rotated_images = []

        print(f'{img_num+1}/{len(images)} processing...')
        #denoised_image = sig(image)
        denoised_image = image

        extended_image = np.zeros((150, 150, 3)).astype(np.uint8)
        for r in range(len(extended_image)):
            for c in range(len(extended_image[r])):
                if 50 <= r < 100 and 50 <= c < 100:
                    extended_image[r][c][0] = denoised_image[r - 50][c - 50][0].copy()
                    extended_image[r][c][1] = denoised_image[r - 50][c - 50][1].copy()
                    extended_image[r][c][2] = denoised_image[r - 50][c - 50][2].copy()

        gaussreg(denoised_image, extended_image)
        spiral_traversal(extended_image, row_begin=49, col_begin=50, row_end=100, col_end=100, window_size=3)

        image_tmp = np.zeros((50, 50))
        for x in range(len(denoised_image[0])):
            for y in range(len(denoised_image)):
                image_tmp[x][y] = denoised_image[x][y][0].copy()
        image_tmp = image_tmp.reshape(-1, 1)

        kmeans = KMeans(n_clusters=2, n_init='auto').fit(image_tmp)
        boundary = kmeans.labels_

        #### boundary check (inversed or not) ####
        avg_vals = {}
        lbs = {}
        for label in boundary:
            if label in lbs:
                lbs[label] += 1
            else:
                lbs[label] = 1
        for px, label in zip(image_tmp, boundary):
            if label in avg_vals:
                avg_vals[label] += px[0]
            else:
                avg_vals[label] = px[0]
        for label in lbs:
            avg_vals[label] /= lbs[label]
        labels = np.array(list(avg_vals.keys()))
        vals = [avg_vals[label] for label in labels]
        sorted_labels = labels[np.argsort(vals)]
        match = {}
        for i, label in enumerate(sorted_labels):
            match[label] = i
        for i in range(len(boundary)):
            boundary[i] = match[boundary[i]]
        boundary = np.array(boundary).astype(np.uint8).reshape(50, 50)
        #############################################

        extended_bd = np.zeros((150, 150, 3)).astype(np.uint8)
        for r in range(len(extended_bd)):
            for c in range(len(extended_bd[r])):
                if 50 <= r < 100 and 50 <= c < 100:
                    extended_bd[r][c][0] = boundary[r - 50][c - 50].copy()
                    extended_bd[r][c][1] = boundary[r - 50][c - 50].copy()
                    extended_bd[r][c][2] = boundary[r - 50][c - 50].copy()

        rot_tmp = rotation(extended_image.copy(), nb=rotation_nb)
        ex_bds = rotation(extended_bd.copy(), nb=rotation_nb)

        for rot_img, rot_bd in zip(rot_tmp, ex_bds):
            rotated_images.append(rot_img[50:100, 50:100].copy())
            min_y = 99999
            max_y = 0
            min_x = 99999
            max_x = 0
            for r, row in enumerate(rot_bd):
                flat_row = [x[0] for x in row]
                if 1 in flat_row:
                    min_y = min(min_y, r)
                if 1 in flat_row and 0 in flat_row:
                    max_y = max(max_y, r)
                for c, col in enumerate(flat_row):
                    if 1 == col:
                        max_x = max(max_x, c)
                        min_x = min(min_x, c)

            shift_mem = [(0, 0)]
            repeat_num = 0
            pass_count = 0

            while repeat_num < shift_nb:
                if 50-min_x >= 100 - max_x:
                    print('type1 err', min_x, max_x, min_y, max_y)
                    min_x = 50
                    max_x = 99
                if 50-min_y >= 100 - max_y:
                    print('type2 err', min_x, max_x, min_y, max_y)
                    min_y = 50
                    max_y = 99
                    #plt.figure()
                    #plt.imshow(boundary)
                    #plt.figure()
                    #plt.imshow(rot_bd*255)
                    #plt.figure()
                    #plt.imshow(rot_img)
                    #plt.show()
                try:
                    x_shift = np.random.randint(50 - min_x, 100 - max_x)
                    y_shift = np.random.randint(50 - min_y, 100 - max_y)
                except:
                    print('catch ', min_x, max_x, min_y, max_y)
                    exit(1)
                if (x_shift, y_shift) in shift_mem:
                    pass_count += 1
                    if pass_count > shift_nb + 100:
                        break
                    continue
                shifted_image = rot_img[
                                (50 - y_shift):(100 - y_shift),
                                (50 - x_shift):(100 - x_shift)
                                ]
                shift_mem.append((x_shift, y_shift))
                shifted_images.append(shifted_image.copy())
                repeat_num += 1

        shifted_images = np.array(shifted_images)
        rotated_images = np.array(rotated_images)

        for i, img in enumerate(rotated_images):
            img = np.array(img).astype(np.uint8)
            iio.imwrite(f'{savepath}/{fname}_{i}_rotated.tif', img)
            if (i+1) % 4 == 0:
                print(f'{i+1}/{len(rotated_images)} completed')

        for i, img in enumerate(shifted_images):
            img = np.array(img).astype(np.uint8)
            iio.imwrite(f'{savepath}/{fname}_{i}_shifted.tif', img)
            if (i+1) % (int((rotation_nb * shift_nb)/10)) == 0:
                print(f'{i+1}/{len(shifted_images)} completed')


def window_avg(matrix, r, c, size=9):
    ll = size // 2
    rr = ll + 1
    std_arr = []
    for row in range(r-ll, r+rr):
        for col in range(c-ll, c+rr):
            if row < 0:
                row = 0
            if col < 0:
                col = 0
            if row >= 150:
                row = 149
            if col >= 150:
                col = 149
            std_arr.append(matrix[row][col][0])
    std = np.std(std_arr)
    noised_val = np.around(max(np.random.normal(np.mean(std_arr), std), np.min(std_arr)))
    matrix[r][c][0] = noised_val
    matrix[r][c][1] = noised_val
    matrix[r][c][2] = noised_val


def addRGBchannel(image):
    new_image = np.zeros((image.shape[0], image.shape[1], 3)).astype(float)
    for r in range(len(image)):
        for c in range(len(image[r])):
            new_image[r][c][0] = image[r][c].copy()
            new_image[r][c][1] = image[r][c].copy()
            new_image[r][c][2] = image[r][c].copy()
            if image[r][c] > 255:
                print(image[r][c])
    return new_image


def projection(image):
    img_arr = iio.imread(image).astype(float)
    projected_matrix = np.zeros((50, 50))
    for i in range(len(img_arr)):
        projected_matrix += (img_arr[i]/65535.) #32773.
    projected_matrix = addRGBchannel(projected_matrix)
    projected_matrix = np.round(projected_matrix * 255).astype(np.uint8)
    return projected_matrix


def quantification(images):
    new_images = []

    for image in images:
        #denoised_image = sig(image).copy()
        denoised_image = image

        image_tmp = np.zeros((denoised_image.shape[0], denoised_image.shape[1]))
        for x in range(len(denoised_image[0])):
            for y in range(len(denoised_image)):
                image_tmp[x][y] = denoised_image[x][y][0].copy()
        image_tmp = image_tmp.reshape(-1, 1)
        clusters = [5, 4, 3, 2, 1, 0]
        nb = 3
        kmeans = KMeans(n_clusters=len(clusters), n_init='auto').fit(image_tmp)
        boundary = kmeans.labels_

        avg_vals = {}
        lbs = {}
        for label in boundary:
            if label in lbs:
                lbs[label] += 1
            else:
                lbs[label] = 1
        for px, label in zip(image_tmp, boundary):
            if label in avg_vals:
                avg_vals[label] += px[0]
            else:
                avg_vals[label] = px[0]
        for label in lbs:
            avg_vals[label] /= lbs[label]
        labels = np.array(list(avg_vals.keys()))
        vals = [avg_vals[label] for label in labels]

        sorted_labels = labels[np.argsort(vals)]

        match = {}
        for i, label in enumerate(sorted_labels):
            match[label] = i
        for i in range(len(boundary)):
            boundary[i] = match[boundary[i]]
        boundary = np.array(boundary).reshape(50, 50)

        for x in range(len(boundary)):
            for y in range(len(boundary[x])):
                if boundary[x][y] in clusters[:nb]:
                    boundary[x][y] = 255//len(clusters) * boundary[x][y]
                else:
                    boundary[x][y] = 0

        image_bd = np.zeros(denoised_image.shape).astype(np.uint8)
        for x in range(len(denoised_image[0])):
            for y in range(len(denoised_image)):
                image_bd[x][y][0] = boundary[x][y].copy()
                image_bd[x][y][1] = boundary[x][y].copy()
                image_bd[x][y][2] = boundary[x][y].copy()
        #image_bd *= 255
        new_images.append(image_bd.copy())
        plt.figure()
        plt.imshow(image)
        plt.figure()
        plt.imshow(image_bd)

        ths_img = thresholding(image, image_bd, spec=3)
        #plt.figure()
        #plt.imshow(ths_img)
        plt.show()

    new_images = np.array(new_images)
    return new_images


def thresholding(image, image_bd, spec=5, coef=1.0001):
    assert spec % 2 != 0

    lighton = []
    new_image = np.zeros(image_bd.shape).astype(np.uint8)
    ll = int(spec/2)
    rr = spec - ll
    for r in range(len(image_bd)):
        for c in range(len(image_bd[r])):
            if image_bd[r][c][0] == 255:
                lighton.append((r, c))

    for r, c in lighton:
        roi_sum = 0
        for roi_r in range(r-ll, r+rr):
            for roi_c in range(c-ll, c+rr):
                if roi_r < 0:
                    roi_r = 0
                if roi_c < 0:
                    roi_c = 0
                if roi_r >= 50:
                    roi_r = 49
                if roi_c >= 50:
                    roi_c = 49

                roi_sum += image[roi_r][roi_c][0]
        roi_sum /= (spec**2)
        if image[r][c][0] > (roi_sum * coef):
            new_image[r][c][0] = image_bd[r][c][0].copy()
            new_image[r][c][1] = image_bd[r][c][1].copy()
            new_image[r][c][2] = image_bd[r][c][2].copy()
    return new_image


def spiral_traversal(matrix, row_begin, col_begin, row_end, col_end, window_size=9):
    start = 1
    res = []
    if len(matrix) == 0:
        return res

    while row_begin >= 0 and col_begin >=0 and row_end < len(matrix) and col_end < len(matrix[0]):
        if start == 1:
            for i in range(col_begin, col_end + 1):
                window_avg(matrix, row_begin, i, window_size)
        else:
            start = 0
            for i in range(col_begin+1, col_end+1):
                window_avg(matrix, row_begin, i, window_size)
        col_begin -= 1
        for i in range(row_begin+1, row_end+1):
            window_avg(matrix, i, col_end, window_size)
        row_begin -= 1
        if row_begin <= row_end:
            for i in range(col_end-1, col_begin-1, -1):
                window_avg(matrix, row_end, i, window_size)
        col_end += 1
        if col_begin <= col_end:
            for i in range(row_end-1, row_begin-1, -1):
                window_avg(matrix, i, col_begin, window_size)
        row_end += 1


def img_save(images, savepath='.'):
    for i, img in enumerate(images):
        img = np.array(img).astype(np.uint8)
        iio.imwrite(f'{savepath}/{i}_aug.tif', img)
        print(f'{i+1}/{len(images)} completed')


"""
comp = './data/selected_samples/1_Stiff Rods/MAX_200514_3880_2h_10_w1GFP-Cam-c-Mos-fast_Roi_5.tif'
data_path2 = './data/selected_raw/1_Stiff Rods/200514_3880_2h_10_w1GFP-Cam-c-Mos-fast_Roi_5.tif'
data_path3 = './data/selected_raw/1_Stiff Rods/200514_3880_2h_06_w1GFP-Cam-c-Mos-fast_Roi_1.tif'
data_path4 = './data/selected_raw/2_Bent rods/200514_3880_2h_02_w1GFP-Cam-c-Mos-fast_Roi_1.tif'
data_path = './data/selected_samples/1_aa'
data_path5 = './data/selected_raw/2_Bent rods/200514_3880_2h_10_w1GFP-Cam-c-Mos-fast_Roi_8.tif'
data_path6 = './data/selected_samples/2_Bent rods/MAX_200514_3880_2h_10_w1GFP-Cam-c-Mos-fast_Roi_8.tif'
data_path7 = './data/selected_samples/2_Bent rods/MAX_200514_3880_2h_10_w1GFP-Cam-c-Mos-fast_Roi_2.tif'
data_path8 = './data/selected_samples/3_Circles_lasso/MAX_200514_3880_4h_09_w1GFP-Cam-c-Mos-fast_Roi_3.tif'
data_path9 = '/Users/junwoopark/Desktop/Junwoo/Faculty/Master/M2/Rad51/data/Rad51_2023_04_06/2h_WT/2_Bent rods/MAX_200515_3880_2h_YPA_01_w1GFP-Cam-c-Mos_Roi_1.tif'
data_path10 = '/Users/junwoopark/Desktop/Junwoo/Faculty/Master/M2/Rad51/data/Rad51_2023_04_06/2h_WT/2_Bent rods/MAX_200514_3880_2h_08_w1GFP-Cam-c-Mos-fast_Roi_6.tif'
"""

"""
#data_path = ['./data/selected_samples/1_Stiff Rods', './data/selected_samples/1_Stiff Rods_shifted']
#data_path = ['./data/selected_samples/2_Bent rods']#, './data/selected_samples/2_Bent rods_shifted']
data_path = ['./data/selected_samples/3_Circles_lasso']#, './data/selected_samples/3_Circles_lasso_shifted']
data_list = data_recur_search(data_path)
for dt in data_list:
    print(dt)
images, labels = imgs_to_ndarray(data_list)
#images = [projection(data_path5)]
images = quantification(images)
#img_save(images, savepath='./data/qt_samples/1_Stiff Rods')
"""




"""go_check = data_path6
plt.figure()
ss = imgs_to_ndarray([go_check])[0][0]/255
plt.imshow(ss)

plt.figure()
ss = imgs_to_ndarray([go_check])[0][0]/255

plt.imshow(sig(ss))

shift([ss], [1])

shift([sig(ss)], [1])
plt.show()
"""
"""
img_arr = iio.imread(data_path5).astype(float)
img_len = len(img_arr)
projected_matrix = np.zeros((50, 50))
for i in range(len(img_arr)):
    projected_matrix += (img_arr[i] / 32773.)  # 32773.   # 65535
    #projected_matrix = sig(projected_matrix)
    #print(list(img_arr[i]))
    print(np.argmax(img_arr[i]))
    kk = np.argmax(img_arr[i])
    r = kk // 50
    c = kk % 50
    plt.figure()
    #plt.imshow(addRGBchannel(sig(projected_matrix)))
    projected_matrix[r][c] += 0.1
    plt.imshow(addRGBchannel(projected_matrix))
    plt.show()

projected_matrix = addRGBchannel(projected_matrix)
plt.figure()
plt.imshow(projected_matrix)

shift([projected_matrix],[1])
plt.show()"""


#data_id = './data/selected_samples/1_Stiff Rods/MAX_210504_3880_6h_06_w1GFP-Cam-c-Mos_Roi_11.tif'
#b = imgs_to_ndarray([data_id])
#rotation_shift(b[0], [data_id], shift_nb=2, rotation_nb=4)

data_list = data_recur_search(['./data/selected_samples/5_others'])
images, labels = imgs_to_ndarray(data_list)
rotation_shift(images, data_list, shift_nb=20, rotation_nb=12, savepath='./data/augmented_data/5_others')
