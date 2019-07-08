import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2
from sklearn.metrics import accuracy_score

from models import generator
from utils import DataLoader, load, save, psnr_error
from constant import const
import app_evaluate as evaluate


slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

num_his = const.NUM_HIS
height, width = 256, 256

snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
evaluate_name = const.EVALUATE

print("This is const = ", const)

def image2_bin(img):
    new_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_image[new_image<30] = 0
    Sum = 0
    count = 0
    for im_col in new_image:
        for im_cel in im_col:
            if im_cel!=0:
                Sum += im_cel
                count+=1
    avg_val = Sum//count
    new_image[new_image<avg_val]=0
    return new_image


# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                             dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    print ("test_outputs: ",test_outputs)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    truth = test_gt
    loss_val = tf.abs((test_outputs - test_gt)*255)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt, dataset_name, evaluate_name):
        load(loader, sess, ckpt)

        psnr_records = []
        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())
        total = 0
        timestamp = time.time()

        
        for video_name, video in videos_info.items():
            length = video['length']
            total += length
            psnrs = np.empty(shape=(length,), dtype=np.float32)

            for i in range(num_his, length):
                video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1)
                psnr = sess.run(test_psnr_error,
                                feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                psnrs[i] = psnr

                print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
                    video_name, num_videos, i, length, psnr))

            psnrs[0:num_his] = psnrs[num_his]
            psnr_records.append(psnrs)
        
        scores = np.array([], dtype=np.float32)
        # video normalization
        for i in range(num_videos):
            distance = psnr_records[i]

            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            # distance = 1 - distance
            scores = np.concatenate((scores, distance[4:]), axis=0)

        used_time = time.time() - timestamp
        print('total time = {}, fps = {}'.format(used_time, total / used_time))
       
        inp_path = '../uploads/'
        out_path = 'frames/'
        
        it = 0
        testPredict = np.zeros(scores.shape, dtype=int)
        thres = 0.6
        
        for video_name, video in videos_info.items():
            length = video['length']
            
            #save_npy_file = 'npy/' + video_name + '.npy'
            dat = np.zeros(length)
            
            for i in range(num_his, length):
                if scores[it] >= thres:
                    k=0
                else:
                    k=1
                testPredict[it] = k
                print('videos = {} / {}, i = {} / {}, Scores = {:.6f}, -{} - {} '.format(
                    video_name, num_videos, i, length, scores[it], 'Abnorm'if k==1 else 'Normal' , k))
                
                dat[i] = scores[it]
                
                # Make output video
                img_path = inp_path + video_name + '/' + '{:03}'.format(i) + '.jpg'
                print ("img paht:", img_path)
                frame_out = out_path + '{:04}'.format(it) + ".jpg"
                frame = cv2.imread(img_path)
                H, W = frame.shape[:2]
                
                video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1)
                l_val = sess.run(loss_val, feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                l_val = np.uint8(l_val)                
                l_val = l_val.reshape(256, 256, 3)                
                l_val = cv2.resize(l_val, (W,H))
                l_val = image2_bin(l_val)
                
                if k==1:
                    cv2.rectangle(frame, (0,0), (W, H), (0, 0, 255), thickness=5, lineType=8, shift=0)
                    frame[l_val!=0] = (0,255,255)
                cv2.imwrite(frame_out, frame)
                
                it = it+1
                
            #np.save(save_npy_file, dat)

    if os.path.isdir(snapshot_dir):
        def check_ckpt_valid(ckpt_name):
            is_valid = False
            ckpt = ''
            if ckpt_name.startswith('model.ckpt-'):
                ckpt_name_splits = ckpt_name.split('.')
                ckpt = str(ckpt_name_splits[0]) + '.' + str(ckpt_name_splits[1])
                ckpt_path = os.path.join(snapshot_dir, ckpt)
                if os.path.exists(ckpt_path + '.index') and os.path.exists(ckpt_path + '.meta') and \
                        os.path.exists(ckpt_path + '.data-00000-of-00001'):
                    is_valid = True

            return is_valid, ckpt

        def scan_psnr_folder():
            tested_ckpt_in_psnr_sets = set()
            for test_psnr in os.listdir(psnr_dir):
                tested_ckpt_in_psnr_sets.add(test_psnr)
            return tested_ckpt_in_psnr_sets

        def scan_model_folder():
            saved_models = set()
            for ckpt_name in os.listdir(snapshot_dir):
                is_valid, ckpt = check_ckpt_valid(ckpt_name)
                if is_valid:
                    saved_models.add(ckpt)
            return saved_models

        tested_ckpt_sets = scan_psnr_folder()
        while True:
            all_model_ckpts = scan_model_folder()
            new_model_ckpts = all_model_ckpts - tested_ckpt_sets

            for ckpt_name in new_model_ckpts:
                # inference
                ckpt = os.path.join(snapshot_dir, ckpt_name)
                inference_func(ckpt, dataset_name, evaluate_name)

                tested_ckpt_sets.add(ckpt_name)

            print('waiting for models...')
            evaluate.evaluate('compute_auc', psnr_dir)
            time.sleep(60)
    else:
        inference_func(snapshot_dir, dataset_name, evaluate_name)
