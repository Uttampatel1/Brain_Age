A:\Uttam\Apps\Brain_Age>python main.py
=== DEBUGGING: Inspecting volume files ===
Inspecting volume files...

=== Folder: 1_40F ===
Files: ['Segments.nii.gz', 'T1w.nii.gz', 'volumes.csv']
Found volume file: volumes.csv
Columns: ['FS Label', 'Structure', 'Volume']
Shape: (96, 3)
First 5 rows:
   FS Label                        Structure       Volume
0         0                       Background  7112.804576
1         2       Left Cortical White Matter   200.935710
2         4           Left Lateral Ventricle    12.201370
3         5  Left Inferior Lateral Ventricle     0.578198
4         7     Left Cerebellar White Matter    12.734652
Data types: FS Label       int64
Structure     object
Volume       float64
dtype: object

=== Folder: 1_40_F ===
Files: ['Segments.nii.gz', 'T1w.nii.gz', 'volumes.csv']
Found volume file: volumes.csv
Columns: ['FS Label', 'Structure', 'Volume']
Shape: (96, 3)
First 5 rows:
   FS Label                        Structure       Volume
0         0                       Background  7406.598498
1         2       Left Cortical White Matter   180.375883
2         4           Left Lateral Ventricle     4.555315
3         5  Left Inferior Lateral Ventricle     0.287842
4         7     Left Cerebellar White Matter    13.670812
Data types: FS Label       int64
Structure     object
Volume       float64
dtype: object

==================================================
=== Running Brain Age Prediction Pipeline ===
Starting Brain Age Prediction Pipeline...
Loaded 192 volumetric features from 1_40F
Loaded 192 volumetric features from 1_40_F
Loaded 192 volumetric features from 1_40_M
Loaded 192 volumetric features from 1_42_M
Loaded data for 4 subjects

⚠️  WARNING: Very small sample size (4 subjects)!
   Brain age prediction models typically require hundreds to thousands of subjects.
   Results should be interpreted with extreme caution and are primarily for demonstration.
Prepared 193 features for 4 subjects
Small dataset: selecting only 3 features to avoid overfitting
Selected 3 features
Dataset size: 4 subjects
Small dataset detected. Using Leave-One-Out cross-validation without test split.
Training models with 4-fold cross-validation...
Random Forest: CV MAE = 0.53 ± 0.85
Gradient Boosting: CV MAE = 0.50 ± 0.86
SVR: CV MAE = 0.67 ± 0.77
Ridge: CV MAE = 0.66 ± 0.78
ElasticNet: CV MAE = 0.84 ± 0.67

Best model: Gradient Boosting
Dataset too small for hyperparameter optimization. Skipping...

Top 3 Most Important Features:
1. vol_Left_parstriangularis: 0.4330
2. vol_fs_1020: 0.3532
3. vol_fs_2006: 0.2138

=== Final Results ===
Best Model: Gradient Boosting
Cross-validation MAE: 0.50 years
Test MAE: 0.50 years
Test R²: nan

