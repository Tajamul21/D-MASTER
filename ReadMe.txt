1. Evaluation for IRCH CView 
Change "evaluate" to "evaluate_csv" for classification datasets [such as IRCH CView] in main.py line 399. 
To save the output csv, change the path in engine.py line 386.
Use evaluation.sh to run the evaluation and generate the csv.
Run match_id_csv_json.py to rename the image_ids to their corresponding names.
Use the output csv file from the previous step in eval_cview_csv.py by choosing the correct "threshold" to generate the scores.

2. Training
Change "evaluate_csv" to "evaluate" for classification datasets [such as IRCH CView] in main.py line 399.
First use the source_only weigths and train the MAE branch using cross_domain_mae.sh. Ensure right keywords for selecting the source and target datasets. Use source_only weights in --resume argument here.
Next use the trained MAE weigths from the previous weights and train the Teacher-student model using teaching.sh. Ensure right keywords for selecting the source and target datasets. Use cross_domain_mae weights in --resume argument here.


