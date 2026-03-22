import argparse
import os
from data_utils.DataLoader import ColoredPointDataset  
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'ComplexRock': [1, 2, 3]}

seg_label_to_cat = {}  # {1:ComplexRock, 2:ComplexRock, 3:ComplexRock}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--color', action='store_true', default=True, help='use color rgb')  # 法向量改为颜色
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

 
    root = 'data/ComplexRock_dataset/'


    TEST_DATASET = ColoredPointDataset(root=root, npoints=args.num_point, split='test', color_channel=args.color)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1  
    num_part = 3  

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, color_channel=args.color).cuda()  
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    '''VISUALIZATION SETUP'''
    output_dir = os.path.join(experiment_dir, 'visualization')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_string(f"Visualization results will be saved to: {os.path.abspath(output_dir)}")

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {1:ComplexRock, 2:ComplexRock, 3:ComplexRock}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
 
        complex_rock_part_ious = [0.0, 0.0, 0.0]
        complex_rock_part_seen = [0, 0, 0]

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

         
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                sample_id = batch_id * args.batch_size + i
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

      
                point_data = points[i].cpu().numpy().transpose(1, 0)  
                gt_label = target[i].reshape(-1, 1)  
                pred_label = cur_pred_val[i].reshape(-1, 1)  

      
                data_to_save = np.hstack([point_data, gt_label, pred_label])

         
                output_path = os.path.join(output_dir, f'{cat}_{sample_id}.txt')
                np.savetxt(
                    output_path,
                    data_to_save,
                    fmt=['%.6f'] * 6 + ['%d', '%d'],  
                    header='x y z r g b gt_label pred_label'  
                )

           
                if cat == 'ComplexRock':
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    for j, l in enumerate(seg_classes[cat]):
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                            part_iou = 1.0
                        else:
                            part_iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                        complex_rock_part_ious[j] += part_iou
                        complex_rock_part_seen[j] += 1

      
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l + 1)  
                total_correct_class[l] += (np.sum((cur_pred_val == l + 1) & (target == l + 1)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

       
        for j in range(3):
            if complex_rock_part_seen[j] > 0:
                complex_rock_part_ious[j] /= complex_rock_part_seen[j]

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])
 
    log_string('ComplexRock Part 1 IoU is: %.5f' % complex_rock_part_ious[0])
    log_string('ComplexRock Part 2 IoU is: %.5f' % complex_rock_part_ious[1])
    log_string('ComplexRock Part 3 IoU is: %.5f' % complex_rock_part_ious[2])


if __name__ == '__main__':
    args = parse_args()
    main(args)
