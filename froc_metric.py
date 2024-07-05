import argparse
import random
import copy
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize
from tqdm import tqdm

def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', default=False, type=bool)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.5, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=40, type=int)
    # Loss coefficients
    parser.add_argument('--teach_box_loss', default=False, type=bool)
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--coef_target', default=1.0, type=float)
    parser.add_argument('--coef_domain', default=1.0, type=float)
    parser.add_argument('--coef_domain_bac', default=0.3, type=float)
    parser.add_argument('--coef_mae', default=1.0, type=float)
    parser.add_argument('--alpha_focal', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.9996, type=float)
    # Dataset parameters
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--source_dataset', default='cityscapes', type=str)
    parser.add_argument('--target_dataset', default='foggy_cityscapes', type=str)
    # Retraining parameters
    parser.add_argument('--epoch_retrain', default=40, type=int)
    parser.add_argument('--keep_modules', default=["decoder"], type=str, nargs="+")
    # MAE parameters
    parser.add_argument('--mae_layers', default=[2], type=int, nargs="+")
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--epoch_mae_decay', default=10, type=float)
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training, "
                             "'cross_domain_mae' for cross domain training with mae, "
                             "'teaching' for teaching process, 'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=8008, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP50', m_ap, epoch)
    for idx, num in zip(idx_to_class.keys(), ap_per_class):
        writer.add_scalar(prefix + '/AP50_%s' % (idx_to_class[idx]['name']), num, epoch)




def get_confmat(pred_list, threshold = 0.1):
    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        box_pascal_gt = ( gt[0]-(gt[2]/2.) , gt[2]-(gt[3]/2.), gt[0]+(gt[2]/2.), gt[1]+(gt[3]/2.) )
        if (pred[0] >= box_pascal_gt[0] and pred[0] <= box_pascal_gt[2] and
                pred[1] >= box_pascal_gt[1] and pred[1] <= box_pascal_gt[3]):
            return True
        return False

    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        for j, gt_box in enumerate(gt_data['boxes']):
            add_tp = False
            new_preds = []
            for pred in pred_boxes:
                if true_positive(gt_box, pred):
                    add_tp = True
                else:
                    new_preds.append(pred)
            pred_boxes = new_preds
            if add_tp:
                out_array[0] += 1
            else:
                out_array[3] += 1
        out_array[2] = len(pred_boxes)
        conf_mat+=out_array
    return conf_mat
    
def calc_froc_threshold(pred_data, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1, 1.5, 2,3,4], num_thresh = 1000):
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat = get_confmat(pred_data, thresh_val)
        conf_mat_thresh[i] = conf_mat
    
    sensitivity = np.zeros((num_thresh)) #recall
    specificity = np.zeros((num_thresh)) #presicion
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if((conf_mat[0]+conf_mat[3])==0):
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[3])
        if((conf_mat[0]+conf_mat[2])==0):
            specificity[i] = 0
        else:
            specificity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[2])

    senses_req = []
    froc_thresh = []
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2]
            if f/num_images < fp_req:
                senses_req.append(sensitivity[i-1])
                froc_thresh.append(thresholds[i])
                print(fp_req, sensitivity[i-1], thresholds[i], get_confmat(pred_data, thresholds[i]))
                break
    return  fps_req, senses_req, froc_thresh, specificity, senses_req,


def calc_froc(pred_data, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1, 1.5, 2,3,4], num_thresh = 1000):
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    conf_mat_thresh = np.zeros((num_thresh, 4))
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat = get_confmat(pred_data, thresh_val)
        conf_mat_thresh[i] = conf_mat
    
    sensitivity = np.zeros((num_thresh)) #recall
    specificity = np.zeros((num_thresh)) #presicion
    for i in range(num_thresh):
        conf_mat = conf_mat_thresh[i]
        if((conf_mat[0]+conf_mat[3])==0):
            sensitivity[i] = 0
        else:
            sensitivity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[3])
        if((conf_mat[0]+conf_mat[2])==0):
            specificity[i] = 0
        else:
            specificity[i] = conf_mat[0]/(conf_mat[0]+conf_mat[2])

    senses_req = []
    for fp_req in fps_req:
        for i in range(num_thresh):
            f = conf_mat_thresh[i][2]
            if f/num_images < fp_req:
                senses_req.append(sensitivity[i-1])
                print(fp_req, sensitivity[i-1], thresholds[i], get_confmat(pred_data, thresholds[i]))
                break
    return senses_req, fps_req, sensitivity, specificity



import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def visualize_and_save_images(data_dict, output_dir):
    for i, sample in tqdm(enumerate(data_dict), total=len(data_dict), desc="Visualization"):
        image = to_pil_image(sample['image'])
        image_np = np.array(image)
        masks = sample['masks']
        target_boxes = sample['target']['boxes']
        pred_boxes = sample['pred']['boxes']

        # Visualize the image with masks, target boxes, and predicted boxes
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np)

        # Plot target boxes
        for box in target_boxes:
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='blue', linewidth=2).set_edgecolor('blue')

        # Plot predicted boxes
        for box in pred_boxes:
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='green', linewidth=2).set_edgecolor('green')

        plt.axis('off')

        # Save the visualization with the original file name
        filename =  f"visualization_{i}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, format=image.format, dpi=image.info.get('dpi'))
        plt.close()


def calc_accuracy(pred_data, num_thresh=100):
    num_images = len(pred_data)
    thresholds = np.linspace(0,1,num_thresh)
    metrics = np.zeros((num_thresh, 2))

    #tp, tn, fp, fn
    for i, thresh_val in enumerate( tqdm(thresholds) ):
        conf_mat= get_confmat_clf(pred_data, thresh_val)
        pres = conf_mat[0]/(conf_mat[0]+conf_mat[2]+ 1) + 0.0001
        recall = conf_mat[0]/(conf_mat[0]+conf_mat[3]+ 1) + 0.0001
        metrics[i,0] = 2*pres*recall/(pres+recall)
        metrics[i,1] = (conf_mat[0]+conf_mat[1])/(conf_mat[0]+conf_mat[1]+conf_mat[2]+conf_mat[3])
       
    max_f1, max_acc = np.argmax(metrics, axis=0)
    print("Max F1 score and Accuracy:", metrics[max_f1], "Threshold:", thresholds[max_f1])
    print("F1 score and Max Accuracy:", metrics[max_acc], "Threshold:", thresholds[max_acc])
    
def get_confmat_clf(pred_list, threshold=0.1):
    #tp, tn, fp, fn
    conf_mat = np.zeros((4))
    conf_mat_idx = []
    for i, data_item in enumerate(pred_list):
        gt_data = data_item['target']
        pred = data_item['pred']
        scores = pred['scores']
        select_mask = scores > threshold
        pred_boxes = pred['boxes'][select_mask]
        out_array = np.zeros((4))
        if(len(gt_data['boxes'])!=0 and len(pred_boxes)!=0):
            out_array[0]+=1
        elif(len(gt_data['boxes'])==0 and len(pred_boxes)!=0):
            out_array[2]+=1
        elif(len(gt_data['boxes'])!=0 and len(pred_boxes)==0):
            out_array[3]+=1
        else:
            out_array[1]+=1
        conf_mat+=out_array
    return conf_mat


# Usage:


def threshold_froc(pred_dict):
    pred_list_test = pred_dict 
    combined_dict = {}

# Iterate over the keys in the dictionaries and combine the values
    for key in pred_list_test[0].keys():
        # Check if the key is 'images' and concatenate the lists
        if key == 'images':
            combined_values = [item for sublist in [d[key] for d in pred_list_test] for item in sublist]
        else:
            # Use list comprehension to extract the values for each key from all dictionaries
            combined_values = [item for sublist in [d[key] for d in pred_list_test] for item in sublist]

        # Assign the combined values directly to the key in the new dictionary
        combined_dict[key] = combined_values
    new_dict = []

# Assuming 'images' is a key in combined_dict
    image_length = len(combined_dict['images'])

    for i in range(image_length):
        new_dict.append({
            'image': combined_dict['images'][i],
            'masks': combined_dict['masks'][i],
            'target': combined_dict['target'][i],
            'pred': combined_dict['pred'][i],
            'image_id' : combined_dict['image_id'][i]
        })

    # import pdb; pdb.set_trace()
    fps_req, sensitivity, thresholds, x, y = calc_froc_threshold(new_dict)
    return fps_req, sensitivity, thresholds




# Evaluate only
def eval_only(model, device):
    if args.distributed:
          Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    # Eval source or target dataset
    
    
    test_loader = build_dataloader(args, args.target_dataset, 'target', 'test', val_trans)
    # test_loader = build_dataloader(args, args.target_dataset, 'source', 'val', val_trans)
    # print(len(val_loader))
    # import pdb; pdb.set_trace()
    
    # pred_list_test = visualize(
    #     model=model,
    #     criterion=criterion,
    #     data_loader_val=test_loader,
    #     output_result_labels=True,
    #     device=device,
    #     print_freq=args.print_freq,
    #     flush=args.flush
    # )
    
    
    pred_list_test = evaluate_froc(
        model=model,
        criterion=criterion,
        data_loader_val=test_loader,
        output_result_labels=True,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )
    

# Create an empty dictionary to store the combined values

    combined_dict = {}

# Iterate over the keys in the dictionaries and combine the values
    for key in pred_list_test[0].keys():
        # Check if the key is 'images' and concatenate the lists
        if key == 'images':
            combined_values = [item for sublist in [d[key] for d in pred_list_test] for item in sublist]
        else:
            # Use list comprehension to extract the values for each key from all dictionaries
            combined_values = [item for sublist in [d[key] for d in pred_list_test] for item in sublist]

        # Assign the combined values directly to the key in the new dictionary
        combined_dict[key] = combined_values
    new_dict = []

# Assuming 'images' is a key in combined_dict
    image_length = len(combined_dict['images'])

    for i in range(image_length):
        new_dict.append({
            'image': combined_dict['images'][i],
            'masks': combined_dict['masks'][i],
            'target': combined_dict['target'][i],
            'pred': combined_dict['pred'][i],
            'image_id' : combined_dict['image_id'][i]
        })

    # import pdb; pdb.set_trace()
    test_froc, test_fpi, test_recall, test_pres = calc_froc(new_dict)
    class_dict = new_dict
    calc_accuracy(class_dict)
    tensor_image = new_dict[0]['image']

    from PIL import Image
    from torchvision.transforms import ToPILImage
    import torchvision.transforms.functional as F
    import numpy as np
    to_pil = ToPILImage()
    pil_image = to_pil(tensor_image.cpu())  # Assuming the tensor is on the CPU

    # Save the image with a file name
    save_path = 'saved_image.png'
    pil_image.save(save_path)
    print("saved")  
       
    print(test_froc, test_fpi)
    # import pdb; 
    # pdb.set_trace()
    # visualize_and_save_images(new_dict, output_dir) 

    # print("Visualizations saved at :", output_dir)  





def main():
    # Initialize distributed mode
    init_distributed_mode(args)
    # Set random seed
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_random_seed(args.random_seed + get_rank())
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    for key, value in args.__dict__.items():
        print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)
    if args.resume != "":
        model = resume_and_load(model, args.resume, device)
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    if args.mode == "single_domain":
        single_domain_training(model, device)
    elif args.mode == "cross_domain_mae":
        cross_domain_mae(model, device)
    elif args.mode == "teaching":
        teaching(model, device)
    elif args.mode == "eval":
        eval_only(model, device)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    # Set output directory
    output_dir = Path(args.output_dir)
    logs_dir = output_dir/'data_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # Call main function
    main()
