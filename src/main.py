from tf2_YOLO.yolov3 import Yolo
from data import split_data

import numpy as np
from tf2_YOLO.utils.kmeans import kmeans, iou_dist, euclidean_dist

from tensorflow.keras.callbacks import LearningRateScheduler

from training import step_scheduler, exp_scheduler, no_scheduler

from tf2_YOLO.utils.tools import get_class_weight
from tensorflow.keras.optimizers.legacy import SGD, Adam

from tf2_YOLO.utils.measurement import create_score_mat
import sys

import wandb
from wandb.keras import WandbMetricsLogger

from tf2_YOLO.utils.measurement import PRfunc

import argparse


# sweep_configuration = {
#     'method':'bayes',
#     'name': 'sweep_yolov3',
#     'metric':{
#         'goal':'maximize',
#         'name':'val/f1_mean'
#     },
#     'parameters':{
#         'k_anchor':{
#             'values': [9, 18, 27]
#         },
#         'weighted_classes':{
#             'values': [True, False]
#         },
#         'dist_function':{
#             'values': ['iou_dist', 'euclidean_dist']
#         },
#         'batch_size':{
#             'values': [4,8,16,32]
#         },
#         'decay':{
#             'values':['none','step','exp']
#         },
#         'lr':{
#             'values': [2.5e-5, 5e-5, 7.5e-5]
#         }, 
#         'optimizer':{
#             'values': ['sgd', 'adam']
#         },
#         'ignore_thresh_loss':{
#             'max': 0.8,
#             'min': 0.2
#         }       
#     }
# }

parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument("--sweep-id", type = str, default = "none")
args = parser.parse_args()

# if(args.sweep_id == 'none'):
#     sweep_id = wandb.sweep(sweep=sweep_configuration, project='sweep_yolov3')
# else:
#     sweep_id = args.sweep_id


def main():
    
    wandb.login()

    wandb.init(project='sweep_yolov3', name='baseline')

    epochs = 50

    # weighted = wandb.config.weighted_classes
    # k_anchor = wandb.config.k_anchor 
    weighted = True
    k_anchor = 9
    # dist_function = iou_dist if wandb.config.dist_function == 'iou_dist' else euclidean_dist
    dist_function = iou_dist 
    
    # ignore_thresh_loss = wandb.config.ignore_thresh_loss 
    ignore_thresh_loss = 0.7
    use_focal_loss = True 
    #batch_size = wandb.config.batch_size 
    batch_size = 5
    #decay = wandb.config.decay
    decay = 'step'
    #lr = wandb.config.lr
    lr = 5e-5
    optimizer = Adam(learning_rate=lr) # if wandb.config.optimizer == 'adam' else SGD(learning_rate=lr, momentum=0.9, decay=5e-4)

    
    if(decay == 'exp'):
        scheduler = exp_scheduler
    elif(decay == 'step'):
        scheduler = step_scheduler
    else:
        scheduler = no_scheduler


    class_names = ["RBC", "WBC", "Platelets"]
    yolo = Yolo(class_names=class_names)
    img_path = "BCCD_Dataset/BCCD/JPEGImages"
    label_path = "BCCD_Dataset/BCCD/Annotations"

    

    print('Loading Dataset')
    train_img, train_label = yolo.read_file_to_dataset(
        img_path,
        label_path,
        shuffle=False,
        thread_num=50)
    
    # Split the Data
    train, val, test = split_data(train_img, train_label)
    train_img, train_labels = train
    val_img, val_labels = val
    test_img, test_labels = test

    
    
    # Get anchor boxes
    
    all_boxes = train_labels[2][train_labels[2][..., 4] == 1][..., 2:4]

    anchors = kmeans(
                all_boxes,
                n_cluster=k_anchor,
                dist_func=dist_function,
                stop_dist=0.000001)

    anchors = np.sort(anchors, axis=0)[::-1][:k_anchor]
    
    yolo.create_model(anchors=anchors)
    #yolo.model.summary()

    callback = LearningRateScheduler(scheduler)

    
    binary_weight_list = []

    if(weighted):
        for i in range(len(train_labels)):
            binary_weight_list.append(
                get_class_weight(
                    train_labels[i][..., 4:5],
                    method='binary'
                )
            )
    else:
        binary_weight_list = [0.1]*3 # [0.1, 0.1, 0.1]


    # Compile model
    loss_weight = {
        "xy":1,
        "wh":1,
        "conf":5,
        "prob":1
    }

    loss = yolo.loss (
        binary_weight_list,
        loss_weight=loss_weight,
        ignore_thresh=ignore_thresh_loss,
        use_focal_loss=use_focal_loss,
    )


    yolo.model.compile(
        optimizer=optimizer,
        loss=loss,
        #metrics=[metrics_obj, metrics_iou, metrics_class, metrics_recall]
        #metrics = metrics
    )
    
    # Training
    train_history = yolo.model.fit(
        train_img,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(val_img, val_labels),
        callbacks=[callback, WandbMetricsLogger()]
    )

    # Testing
    prediction = yolo.model.predict(test_img, batch_size=10)
    score_df = create_score_mat(
        test_labels[2],
        prediction[2],
        prediction[1],
        prediction[0],
        class_names=class_names,
        conf_threshold=0.5,
        nms_mode=2,
        nms_threshold=0.5,
        version=3)
    
    #print(score_df)
    
    f1_mean = sum(score_df['F1-score'])/len(score_df['F1-score'])

    table = wandb.Table(dataframe=score_df)
    wandb.log({"test/score_metrics": table, "test/f1_mean": f1_mean})
    
    
    prediction = yolo.model.predict(val_img, batch_size=10)
    score_df = create_score_mat(
        val_labels[2],
        prediction[2],
        prediction[1],
        prediction[0],
        class_names=class_names,
        conf_threshold=0.5,
        nms_mode=2,
        nms_threshold=0.5,
        version=3)
    
    #print(score_df)
    
    f1_mean = sum(score_df['F1-score'])/len(score_df['F1-score'])

    table = wandb.Table(dataframe=score_df)
    wandb.log({"val/score_metrics": table, "val/f1_mean": f1_mean})


# Start sweep job.
#wandb.agent(sweep_id, function=main)


if __name__ == '__main__':
    main()

