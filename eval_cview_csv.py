import numpy as np
import shutil
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def create_label_map(gt_file, pred_file):
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    merged = gt_df.merge(pred_df, on='img_path', how='inner')
    
    mal_logits = merged['mal_score'].to_numpy()
    true_labels = merged['label'].to_numpy()
    return true_labels, mal_logits

def create_label_map_2(gt_file, pred_file):
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    merged = gt_df.merge(pred_df, on='img_path', how='inner')
    # import pdb; pdb.set_trace()
    
    mal_logits = merged['mal_score'].to_numpy()
    true_labels = merged['label'].to_numpy()
    img_paths = merged['img_path'].to_numpy()
    return img_paths, true_labels, mal_logits
    
    
def calc_metrics(true_labels, mal_logits, threshold, filename):
    predictions = np.zeros_like(true_labels)    
    for i in range(len(mal_logits)):
        if(mal_logits[i]>threshold):
            predictions[i]=1
        
    # print(classification_report(true_labels, predictions, labels=[0, 1]))
    auc_score = roc_auc_score(true_labels, mal_logits)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    npv = tn/(tn+fn)
    print(filename)
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print("AUC score", auc_score, "NPV score", npv)
    conf_mat = [str(item) for item in [tn, fp, fn, tp]]
    
    # file = open(filename, 'w')
    # file.write(classification_report(true_labels, predictions, labels=[0, 1]))
    # file.write(f'Confusion Matrix: tn, fp, fn, tp {" ".join(conf_mat)}\n')
    # file.write(f'AUC_Score: {auc_score:.4f}\n')
    # file.write(f'NPV_Score: {npv:.4f}\n')
    # file.close()
    
    return predictions


def check_same(arr1, arr2):
    for i,val in enumerate(arr2):
        if(arr2[i]!=val):
            print("ERROR in image order")
            exit(0)
            return None
    return


def save_fn(preds, true_labels, img_paths, data_folder):
    # import pdb; pdb.set_trace()
    preds_n = np.where(np.array(preds) == 0)
    true_n = np.where(np.array(true_labels) == 0)
    fn_idxs = np.setdiff1d(preds_n[0], true_n[0])
    for i,fn_idx in enumerate(fn_idxs):
        img_path = os.path.join(data_folder, img_paths[fn_idx])
        trgt_path = os.path.join("", img_paths[fn_idx])
        shutil.copy(img_path, trgt_path) 
     

if __name__=='__main__':
    # Sensitivity 0.95 
    # threshold = 0.0410 #focalnet
    # threshold = 0.003 #smallmass
    # threshold = 0.128 #densemass

    # Sensitivity 0.90
    # threshold = 0.087 #focalnet
    # threshold = 0.03 #smallmass
    # threshold = 0.3 #densemass
    
    # F1-score optimum (newly trained)
    # threshold = 0.028 # focalnet
    # threshold = 0.587 # cen
    # threshold = 0.567 # smallmass
    # threshold = 0.156 # history
    thresholds = {
        # "focalnet": 0.028,
        # "cen" : 0.587,
        # "smallmass": 0.567,
        # "history": 0.156,
        # "densemass": 0.55,
        # "dmaster_source": 0.250,
        # "dmaster_adapt_cross_domain": 0.219,
        "dmaster_adapt_best": 0.221,
        # "dmaster_adapt_best": 0.50, #Trying out higher confidences
        # "dmaster_adapt_tch": 0.240
    }
    
    # F1-score optimum (previously trained)
    # threshold = 0.321 # focalnet
    # threshold = 0.632 # cen
    # threshold = 0.353 # smallmass
    # threshold = 0.030441519 # history


    # # F1-score optimum (previously trained)
    # threshold = 0.004754 # focalnet
    # threshold = 0.502059 # cen
    # threshold = 0.005219 # smallmass
    # threshold = 0.030441519 # history
    # thresholds = {
    #     "focalnet": 0.004754,
    #     "cen" : 0.502059,
    #     "smallmass": 0.005219,
    #     "history": 0.030441519
    # }

    # gt_file = "./data/irch_gt.csv"
    # pred_file = f"./preds_new/smallmass_preds.csv"
    gt_file = "/home/kaustubh/scratch/Mammo_Datasets_negroni/Dmaster_Data/c_view_data/irch_gt.csv"
    # gt_file = "./data_cview/irch_subset_gt.csv"
    pred_file = f"/home/kaustubh/scratch/D-MASTER/outputs_krb/teaching/csv_preds_kaustubh/1_d_master_adap_cview_post_training_with_names.csv"
    img_paths_true, true_labels_true, img_paths= create_label_map_2(gt_file, pred_file)
    predictions = []
    for i,(model_name, threshold) in enumerate(thresholds.items()):
        # pred_file = f"./cview_preds/{model_name}_preds.csv"
        print(pred_file)
        # pred_file = f"./preds_new/{model_name}_preds.csv"
        img_paths, true_labels, mal_logits = create_label_map_2(gt_file, pred_file)
        check_same(img_paths_true, img_paths)
        check_same(true_labels_true, true_labels)
        model_predictions = calc_metrics(true_labels, mal_logits, threshold, model_name+"_metrics.txt")
        predictions.append(model_predictions)
    
    # import pdb; pdb.set_trace() 
    predictions = np.array(predictions)
    preds = np.max(predictions, axis=0)
    
    print(classification_report(true_labels_true, preds, labels=[0, 1]))
    tn, fp, fn, tp = confusion_matrix(true_labels_true, preds).ravel()
    npv = tn/(tn+fn)
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print("NPV score", npv)
    # save_fn(preds, true_labels_true, img_paths, "/home/kshitiz/scratch/FocalNet-DINO/MULTI_MODEL_DATA/IRCH_DATA/Mammo_PNG")
    # save_fn(preds, true_labels_true, img_paths, "/home/tajamul/scratch/DA/DATA/Dmaster_Data/c_view_data/common_cview")
