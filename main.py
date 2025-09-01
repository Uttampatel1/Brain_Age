import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class BrainAgePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.features = None
        self.targets = None
        self.scaler = StandardScaler()
        self.best_model = None
        
    def parse_folder_info(self, folder_name):
        """Extract age and gender from folder names like 1_40_M, 1_42_F, etc."""
        # Pattern to match folder names with age and gender
        pattern = r'(\d+)_(\d+)_?([MF])'
        match = re.search(pattern, folder_name)
        if match:
            subject_id = match.group(1)
            age = int(match.group(2))
            gender = match.group(3)
            return subject_id, age, gender
        return None, None, None
    
    def inspect_volume_files(self):
        """Debug function to inspect the format of volume files"""
        print("Inspecting volume files...")
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            print(f"\n=== Folder: {folder} ===")
            
            # List all files in folder
            files = os.listdir(folder_path)
            print(f"Files: {files}")
            
            # Look for volume file
            volume_file = None
            for file in files:
                if 'volume' in file.lower() or file.endswith('.csv') or file.endswith('.txt') or file.endswith('.xlsx'):
                    volume_file = os.path.join(folder_path, file)
                    print(f"Found volume file: {file}")
                    
                    try:
                        # Try to read and display first few rows
                        if volume_file.endswith('.xlsx') or volume_file.endswith('.xls'):
                            df = pd.read_excel(volume_file)
                        elif volume_file.endswith('.csv'):
                            df = pd.read_csv(volume_file)
                        else:
                            df = pd.read_csv(volume_file, sep=None, engine='python')
                        
                        print(f"Columns: {list(df.columns)}")
                        print(f"Shape: {df.shape}")
                        print("First 5 rows:")
                        print(df.head())
                        
                        # Show data types
                        print(f"Data types: {df.dtypes}")
                        
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
                    
                    break
            
            if volume_file is None:
                print("No volume file found")
            
            # Only inspect first 2 folders for brevity
            if len([f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]) > 1:
                if folder == sorted([f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))])[1]:
                    break
    
    def load_volumetric_features(self):
        """Load volumetric features from FreeSurfer segmentations"""
        data_list = []
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            subject_id, age, gender = self.parse_folder_info(folder)
            if age is None:
                continue
                
            # Look for volume file (Excel or text file)
            volume_file = None
            for file in os.listdir(folder_path):
                if 'volume' in file.lower() or file.endswith('.csv') or file.endswith('.txt'):
                    volume_file = os.path.join(folder_path, file)
                    break
            
            if volume_file:
                try:
                    # Try to read volume data
                    if volume_file.endswith('.xlsx') or volume_file.endswith('.xls'):
                        volumes = pd.read_excel(volume_file)
                    elif volume_file.endswith('.csv'):
                        volumes = pd.read_csv(volume_file)
                    else:
                        # Try reading as space/tab separated text file
                        volumes = pd.read_csv(volume_file, sep=None, engine='python')
                    
                    # Create feature dictionary
                    feature_dict = {
                        'subject_id': subject_id,
                        'age': age,
                        'gender': 1 if gender == 'M' else 0,
                        'folder_name': folder
                    }
                    
                    # Handle FreeSurfer format: FS_Label, Structure_Name, Volume
                    if len(volumes.columns) >= 3:
                        # 3-column format: Label, Name, Volume
                        for idx, row in volumes.iterrows():
                            try:
                                fs_label = row.iloc[0]
                                region_name = str(row.iloc[1]).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                                volume_value = float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0
                                
                                # Clean region name
                                region_name = region_name.replace('__', '_').strip('_')
                                feature_dict[f'vol_{region_name}'] = volume_value
                                
                                # Also add by FS label if numeric
                                if pd.notna(fs_label) and str(fs_label).isdigit():
                                    feature_dict[f'vol_fs_{fs_label}'] = volume_value
                            except (ValueError, IndexError) as e:
                                continue  # Skip problematic rows
                    
                    elif len(volumes.columns) >= 2:
                        # 2-column format: Name, Volume
                        for idx, row in volumes.iterrows():
                            try:
                                region_name = str(row.iloc[0]).replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                                volume_value = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0
                                region_name = region_name.replace('__', '_').strip('_')
                                feature_dict[f'vol_{region_name}'] = volume_value
                            except (ValueError, IndexError) as e:
                                continue  # Skip problematic rows
                    
                    # Only add if we got some volumetric features
                    vol_features = [k for k in feature_dict.keys() if k.startswith('vol_')]
                    if vol_features:
                        data_list.append(feature_dict)
                        print(f"Loaded {len(vol_features)} volumetric features from {folder}")
                    else:
                        print(f"No volumetric features found in {folder}")
                    
                except Exception as e:
                    print(f"Error loading volumes from {folder}: {e}")
                    
            # Alternative: Extract features from segmentation file
            seg_file = os.path.join(folder_path, 'Segments.nii.gz')
            if os.path.exists(seg_file) and volume_file is None:
                try:
                    seg_features = self.extract_segmentation_features(seg_file)
                    feature_dict = {
                        'subject_id': subject_id,
                        'age': age,
                        'gender': 1 if gender == 'M' else 0,
                        'folder_name': folder
                    }
                    feature_dict.update(seg_features)
                    data_list.append(feature_dict)
                except Exception as e:
                    print(f"Error processing segmentation from {folder}: {e}")
        
        if data_list:
            self.df = pd.DataFrame(data_list)
            print(f"Loaded data for {len(self.df)} subjects")
            return True
        return False
    
    def load_freesurfer_stats(self):
        """Load volumetric features from FreeSurfer stats files"""
        data_list = []
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            subject_id, age, gender = self.parse_folder_info(folder)
            if age is None:
                continue
            
            # Look for FreeSurfer stats files
            stats_files = []
            for file in os.listdir(folder_path):
                if file.endswith('.stats') or 'aseg' in file or 'aparc' in file:
                    stats_files.append(os.path.join(folder_path, file))
            
            feature_dict = {
                'subject_id': subject_id,
                'age': age,
                'gender': 1 if gender == 'M' else 0,
                'folder_name': folder
            }
            
            # Parse each stats file
            for stats_file in stats_files:
                try:
                    with open(stats_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('#') or not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:  # Standard FreeSurfer format
                            try:
                                label_id = parts[0]
                                structure_name = parts[4].replace('-', '_').replace(' ', '_')
                                volume = float(parts[3])
                                
                                feature_dict[f'vol_{structure_name}'] = volume
                                if label_id.isdigit():
                                    feature_dict[f'vol_fs_{label_id}'] = volume
                                    
                            except (ValueError, IndexError):
                                continue
                                
                except Exception as e:
                    print(f"Error reading stats file {stats_file}: {e}")
            
            # Only add if we got volumetric features
            vol_features = [k for k in feature_dict.keys() if k.startswith('vol_')]
            if vol_features:
                data_list.append(feature_dict)
                print(f"Loaded {len(vol_features)} features from {folder}")
        
        if data_list:
            self.df = pd.DataFrame(data_list)
            print(f"Loaded FreeSurfer data for {len(self.df)} subjects")
            return True
        return False

    def create_manual_volume_data(self):
        """Create volume data from the provided FreeSurfer statistics"""
        # Your provided FreeSurfer statistics
        fs_data = {
            'Background': 11242.44,
            'Left_Cortical_White_Matter': 231.8224,
            'Left_Lateral_Ventricle': 5.493147,
            'Left_Inferior_Lateral_Ventricle': 0.282489,
            'Left_Cerebellar_White_Matter': 13.08721,
            'Left_Cerebellar_Cortex': 46.04806,
            'Left_Thalamus': 7.166775,
            'Left_Caudate': 2.903008,
            'Left_Putamen': 4.094511,
            'Left_Pallidum': 1.908378,
            'rd_Ventricle': 1.200972,
            'th_Ventricle': 1.431382,
            'Brain_Stem': 18.51447,
            'Left_Hippocampus': 4.01876,
            'Left_Amygdala': 1.734782,
            'CSF': 0.91296,
            'Left_Accumbens': 0.367315,
            'Left_Ventral_DC': 4.17263,
            'Left_Choroid_Plexus': 0.267891,
            'Right_Cortical_White_Matter': 233.1086,
            'Right_Lateral_Ventricle': 6.340219,
            'Right_Inferior_Lateral_Ventricle': 0.14519,
            'Right_Cerebellar_White_Matter': 12.98936,
            'Right_Cerebellar_Cortex': 48.19079,
            'Right_Thalamus': 6.563922,
            'Right_Caudate': 2.944829,
            'Right_Putamen': 4.144223,
            'Right_Pallidum': 1.943098,
            'Right_Hippocampus': 4.332812,
            'Right_Amygdala': 1.755298,
            'Right_Accumbens': 0.503825,
            'Right_Ventral_DC': 4.410536,
            'Right_Choroid_Plexus': 0.490805,
            'WM_hypointensities': 1.465313,
            'Left_caudalanteriorcingulate': 2.845405,
            'Left_caudalmiddlefrontal': 6.40374,
            'Left_cuneus': 3.829382,
            'Left_entorhinal': 2.352628,
            'Left_fusiform': 6.83418,
            'Left_inferiorparietal': 11.25261,
            'Left_inferiortemporal': 12.33759,
            'Left_isthmuscingulate': 2.347893,
            'Left_lateraloccipital': 14.37458,
            'Left_lateralorbitofrontal': 8.746504,
            'Left_lingual': 4.912388,
            'Left_medialorbitofrontal': 4.369898,
            'Left_middletemporal': 11.18238,
            'Left_parahippocampal': 1.91114,
            'Left_paracentral': 3.815968,
            'Left_parsopercularis': 4.958154,
            'Left_parsorbitalis': 2.366437,
            'Left_parstriangularis': 4.835453,
            'Left_pericalcarine': 0.845494,
            'Left_postcentral': 9.391573,
            'Left_posteriorcingulate': 3.790323,
            'Left_precentral': 12.21094,
            'Left_precuneus': 8.629721,
            'Left_rostralanteriorcingulate': 3.303858,
            'Left_rostralmiddlefrontal': 10.6032,
            'Left_superiorfrontal': 22.2531,
            'Left_superiorparietal': 9.444836,
            'Left_superiortemporal': 15.25243,
            'Left_supramarginal': 8.601709,
            'Left_transversetemporal': 1.157179,
            'Left_insula': 5.793785,
            'Right_caudalanteriorcingulate': 2.005435,
            'Right_caudalmiddlefrontal': 6.472784,
            'Right_cuneus': 2.859609,
            'Right_entorhinal': 2.298971,
            'Right_fusiform': 7.025926,
            'Right_inferiorparietal': 13.01146,
            'Right_inferiortemporal': 13.55592,
            'Right_isthmuscingulate': 1.924554,
            'Right_lateraloccipital': 9.701285,
            'Right_lateralorbitofrontal': 9.015184,
            'Right_lingual': 4.468533,
            'Right_medialorbitofrontal': 4.318609,
            'Right_middletemporal': 11.78168,
            'Right_parahippocampal': 1.974266,
            'Right_paracentral': 4.04046,
            'Right_parsopercularis': 4.63345,
            'Right_parsorbitalis': 2.33803,
            'Right_parstriangularis': 4.588078,
            'Right_pericalcarine': 1.326435,
            'Right_postcentral': 9.386049,
            'Right_posteriorcingulate': 3.254541,
            'Right_precentral': 12.1273,
            'Right_precuneus': 10.19327,
            'Right_rostralanteriorcingulate': 2.448106,
            'Right_rostralmiddlefrontal': 10.05203,
            'Right_superiorfrontal': 25.7278,
            'Right_superiorparietal': 11.42739,
            'Right_superiortemporal': 16.31018,
            'Right_supramarginal': 8.943773,
            'Right_transversetemporal': 1.019091,
            'Right_insula': 6.218307
        }
        
        # Create sample data for demonstration
        data_list = []
        
        # Create synthetic subjects with age variations
        base_ages = [40, 41, 42]  # Based on your folder names
        genders = ['M', 'F']
        
        subject_id = 1
        for age in base_ages:
            for gender in genders:
                for variation in range(2):  # Create 2 variations per age/gender
                    # Add some realistic variation to the volumes
                    noise_factor = 1 + np.random.normal(0, 0.1)  # 10% variation
                    
                    feature_dict = {
                        'subject_id': f"{subject_id}_{age}_{gender}_{variation}",
                        'age': age + np.random.randint(-2, 3),  # Age variation
                        'gender': 1 if gender == 'M' else 0,
                        'folder_name': f"{subject_id}_{age}_{gender}_{variation}"
                    }
                    
                    # Add all volumetric features with variation
                    for region, volume in fs_data.items():
                        feature_dict[f'vol_{region}'] = volume * noise_factor * (1 + np.random.normal(0, 0.05))
                    
                    data_list.append(feature_dict)
                    subject_id += 1
        
        if data_list:
            self.df = pd.DataFrame(data_list)
            print(f"Created synthetic data for {len(self.df)} subjects with {len(fs_data)} volumetric features")
            return True
        return False

    def extract_segmentation_features(self, seg_file):
        """Extract volumetric features from segmentation file"""
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        
        features = {}
        unique_labels = np.unique(seg_data)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            mask = seg_data == label
            volume = np.sum(mask)
            features[f'vol_region_{int(label)}'] = volume
            
        return features
    
    def load_image_features(self):
        """Load and extract features from T1w images"""
        data_list = []
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            subject_id, age, gender = self.parse_folder_info(folder)
            if age is None:
                continue
                
            t1_file = os.path.join(folder_path, 'T1w.nii.gz')
            if os.path.exists(t1_file):
                try:
                    features = self.extract_t1_features(t1_file)
                    feature_dict = {
                        'subject_id': subject_id,
                        'age': age,
                        'gender': 1 if gender == 'M' else 0,
                        'folder_name': folder
                    }
                    feature_dict.update(features)
                    data_list.append(feature_dict)
                    
                except Exception as e:
                    print(f"Error processing T1 image from {folder}: {e}")
        
        if data_list:
            self.df = pd.DataFrame(data_list)
            print(f"Loaded T1 features for {len(self.df)} subjects")
            return True
        return False
    
    def extract_t1_features(self, t1_file):
        """Extract statistical features from T1w images"""
        img = nib.load(t1_file)
        data = img.get_fdata()
        
        # Basic statistical features
        features = {
            'mean_intensity': np.mean(data[data > 0]),
            'std_intensity': np.std(data[data > 0]),
            'median_intensity': np.median(data[data > 0]),
            'max_intensity': np.max(data),
            'min_intensity': np.min(data[data > 0]),
            'volume': np.sum(data > 0),
            'q25_intensity': np.percentile(data[data > 0], 25),
            'q75_intensity': np.percentile(data[data > 0], 75),
        }
        
        # Texture features (simplified)
        # Gray level co-occurrence matrix features could be added here
        gradient = np.gradient(data)
        features['gradient_mean'] = np.mean([np.mean(g) for g in gradient])
        features['gradient_std'] = np.std([np.std(g) for g in gradient])
        
        return features
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        if self.df is None:
            print("No data loaded!")
            return False
            
        # Separate features and target
        feature_cols = [col for col in self.df.columns 
                       if col not in ['subject_id', 'age', 'folder_name']]
        
        self.features = self.df[feature_cols].fillna(0)
        self.targets = self.df['age'].values
        
        print(f"Prepared {self.features.shape[1]} features for {len(self.targets)} subjects")
        return True
    
    def feature_selection(self, k=50):
        """Select top k features using univariate selection"""
        if self.features is None:
            return False
        
        n_samples = len(self.targets)
        max_features = min(k, self.features.shape[1])
        
        # For very small datasets, use fewer features to avoid overfitting
        if n_samples < 10:
            max_features = min(max_features, n_samples - 1)
            print(f"Small dataset: selecting only {max_features} features to avoid overfitting")
        elif n_samples < 20:
            max_features = min(max_features, n_samples // 2)
            print(f"Small dataset: selecting {max_features} features")
        
        if max_features <= 0:
            print("Dataset too small for feature selection. Using all features.")
            return True
            
        selector = SelectKBest(score_func=f_regression, k=max_features)
        self.features = pd.DataFrame(
            selector.fit_transform(self.features, self.targets),
            columns=[self.features.columns[i] for i in selector.get_support(indices=True)]
        )
        
        print(f"Selected {self.features.shape[1]} features")
        return True
    
    def train_models(self):
        """Train multiple models and select the best one"""
        if self.features is None or self.targets is None:
            print("Features not prepared!")
            return False
        
        n_samples = len(self.targets)
        print(f"Dataset size: {n_samples} subjects")
        
        # Adjust strategy based on dataset size
        if n_samples < 10:
            # For very small datasets, use leave-one-out and no test split
            print("Small dataset detected. Using Leave-One-Out cross-validation without test split.")
            X_train_scaled = self.scaler.fit_transform(self.features)
            y_train = self.targets
            use_test_split = False
            cv_folds = n_samples  # Leave-One-Out
        elif n_samples < 20:
            # For small datasets, use smaller test split and fewer CV folds
            print("Small dataset detected. Using smaller test split and 3-fold CV.")
            test_size = max(0.1, 1/n_samples)  # At least 1 sample for test
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.targets, test_size=test_size, random_state=42
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            use_test_split = True
            cv_folds = 3
        else:
            # Standard approach for larger datasets
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.targets, test_size=0.2, random_state=42
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            use_test_split = True
            cv_folds = 5
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),  # Reduced trees for small data
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),  # Reduced trees
            'SVR': SVR(kernel='rbf'),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42)
        }
        
        results = {}
        
        print(f"Training models with {cv_folds}-fold cross-validation...")
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=cv_folds, scoring='neg_mean_absolute_error')
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                if use_test_split:
                    # Test predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'model': model,
                        'cv_mae': -cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_mae': mae,
                        'test_mse': mse,
                        'test_r2': r2,
                        'predictions': y_pred,
                        'actual': y_test
                    }
                    
                    print(f"{name}: CV MAE = {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}, "
                          f"Test MAE = {mae:.2f}, R² = {r2:.3f}")
                else:
                    # No test split - use CV scores only
                    results[name] = {
                        'model': model,
                        'cv_mae': -cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_mae': -cv_scores.mean(),  # Use CV as proxy
                        'test_mse': np.nan,
                        'test_r2': np.nan,
                        'predictions': np.array([]),
                        'actual': np.array([])
                    }
                    
                    print(f"{name}: CV MAE = {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not results:
            print("No models could be trained successfully!")
            return False
        
        # Select best model based on cross-validation MAE
        best_model_name = min(results.keys(), key=lambda x: results[x]['cv_mae'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        
        return results
    
    def optimize_best_model(self):
        """Hyperparameter optimization for the best model"""
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        n_samples = len(self.targets)
        
        # Skip hyperparameter optimization for very small datasets
        if n_samples < 10:
            print("Dataset too small for hyperparameter optimization. Skipping...")
            return
        
        # Use appropriate CV folds and test split based on dataset size
        if n_samples < 20:
            test_size = max(0.1, 1/n_samples)
            cv_folds = 3
        else:
            test_size = 0.2
            cv_folds = 5
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Define simpler parameter grids for small datasets
        if isinstance(self.best_model, RandomForestRegressor):
            if n_samples < 20:
                param_grid = {
                    'n_estimators': [25, 50],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 3]
                }
            else:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
        elif isinstance(self.best_model, GradientBoostingRegressor):
            if n_samples < 20:
                param_grid = {
                    'n_estimators': [25, 50],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            else:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
        elif isinstance(self.best_model, SVR):
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        else:
            print("No hyperparameter optimization available for this model")
            return
        
        print("Optimizing hyperparameters...")
        try:
            grid_search = GridSearchCV(
                self.best_model, param_grid, cv=cv_folds, 
                scoring='neg_mean_absolute_error', n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.best_model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV MAE: {-grid_search.best_score_:.2f}")
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            print("Continuing with default parameters...")
    
    def plot_results(self, results):
        """Plot model performance and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_names = list(results.keys())
        cv_maes = [results[name]['cv_mae'] for name in model_names]
        test_maes = [results[name]['test_mae'] for name in model_names]
        
        x_pos = range(len(model_names))
        width = 0.35
        
        axes[0, 0].bar([x - width/2 for x in x_pos], cv_maes, width, alpha=0.7, label='CV MAE')
        axes[0, 0].bar([x + width/2 for x in x_pos], test_maes, width, alpha=0.7, label='Test MAE')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Best model predictions vs actual (if test predictions exist)
        best_name = min(results.keys(), key=lambda x: results[x]['cv_mae'])
        y_pred = results[best_name]['predictions']
        y_actual = results[best_name]['actual']
        
        if len(y_pred) > 0 and len(y_actual) > 0:
            axes[0, 1].scatter(y_actual, y_pred, alpha=0.6, s=100)
            axes[0, 1].plot([y_actual.min(), y_actual.max()], 
                           [y_actual.min(), y_actual.max()], 'r--', lw=2)
            axes[0, 1].set_xlabel('Actual Age')
            axes[0, 1].set_ylabel('Predicted Age')
            axes[0, 1].set_title(f'Best Model ({best_name}) Predictions')
            
            # Add correlation coefficient
            corr = np.corrcoef(y_actual, y_pred)[0, 1]
            axes[0, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 1].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[0, 1].text(0.5, 0.5, 'No test predictions\n(dataset too small)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, 1].set_title(f'Best Model: {best_name}')
        
        # Residuals plot (if test predictions exist)
        if len(y_pred) > 0 and len(y_actual) > 0:
            residuals = y_actual - y_pred
            axes[1, 0].scatter(y_pred, residuals, alpha=0.6, s=100)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Predicted Age')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residuals Plot')
        else:
            axes[1, 0].text(0.5, 0.5, 'No residuals plot\n(no test predictions)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 0].set_title('Residuals Plot')
        
        # Age distribution
        ages = self.targets
        axes[1, 1].hist(ages, bins=min(10, len(ages)), alpha=0.7, label='Actual Ages', edgecolor='black')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Age Distribution (n={len(ages)})')
        axes[1, 1].legend()
        
        # Add dataset info
        axes[1, 1].text(0.05, 0.95, f'Age range: {ages.min():.1f}-{ages.max():.1f}\nMean: {ages.mean():.1f}±{ages.std():.1f}', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.best_model, 'feature_importances_'):
            return
        
        importances = self.best_model.feature_importances_
        feature_names = self.features.columns
        
        # Adjust top_n to actual number of features available
        actual_top_n = min(top_n, len(importances))
        
        indices = np.argsort(importances)[::-1][:actual_top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {actual_top_n} Feature Importances')
        plt.bar(range(actual_top_n), importances[indices])
        plt.xticks(range(actual_top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Feature Importance')
        plt.xlabel('Brain Regions')
        plt.tight_layout()
        plt.show()
        
        # Print feature importance values
        print(f"\nTop {actual_top_n} Most Important Features:")
        for i in range(actual_top_n):
            idx = indices[i]
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict_age(self, new_features):
        """Predict age for new data"""
        if self.best_model is None:
            print("No model trained!")
            return None
        
        new_features_scaled = self.scaler.transform(new_features)
        prediction = self.best_model.predict(new_features_scaled)
        return prediction
    
    def run_full_pipeline(self, feature_type='volumetric', use_feature_selection=True, use_manual_data=False):
        """Run the complete brain age prediction pipeline"""
        print("Starting Brain Age Prediction Pipeline...")
        
        # Load data
        if use_manual_data:
            success = self.create_manual_volume_data()
        elif feature_type == 'volumetric':
            success = self.load_volumetric_features()
            if not success:
                print("Trying FreeSurfer stats files...")
                success = self.load_freesurfer_stats()
        else:
            success = self.load_image_features()
        
        if not success:
            print("Failed to load data!")
            return None
        
        # Check sample size and warn if too small
        n_samples = len(self.df)
        if n_samples < 10:
            print(f"\n⚠️  WARNING: Very small sample size ({n_samples} subjects)!")
            print("   Brain age prediction models typically require hundreds to thousands of subjects.")
            print("   Results should be interpreted with extreme caution and are primarily for demonstration.")
        elif n_samples < 50:
            print(f"\n⚠️  WARNING: Small sample size ({n_samples} subjects)!")
            print("   Consider collecting more data for reliable brain age prediction.")
        
        # Prepare features
        self.prepare_features()
        
        # Feature selection
        if use_feature_selection and self.features.shape[1] > 10:
            self.feature_selection(k=min(50, self.features.shape[1] // 2))
        
        # Train models
        results = self.train_models()
        
        if results:
            # Optimize best model (skip for very small datasets)
            self.optimize_best_model()
            
            # Plot results
            self.plot_results(results)
        
        return results

# Usage example
if __name__ == "__main__":
    # Set your data path
    data_path = r"A:\Uttam\Apps\Brain_Age\output"  # Updated with your path
    
    # Create predictor instance
    predictor = BrainAgePredictor(data_path)
    
    # First, inspect the volume files to understand the format
    print("=== DEBUGGING: Inspecting volume files ===")
    predictor.inspect_volume_files()
    
    print("\n" + "="*50)
    print("=== Running Brain Age Prediction Pipeline ===")
    
    # Run the complete pipeline
    results = predictor.run_full_pipeline(
        feature_type='volumetric',  # or 'image' for T1w features
        use_feature_selection=True
    )
    
    # Print final results
    if results:
        best_model = min(results.keys(), key=lambda x: results[x]['cv_mae'])
        print(f"\n=== Final Results ===")
        print(f"Best Model: {best_model}")
        print(f"Cross-validation MAE: {results[best_model]['cv_mae']:.2f} years")
        print(f"Test MAE: {results[best_model]['test_mae']:.2f} years")
        print(f"Test R²: {results[best_model]['test_r2']:.3f}")
    else:
        print("\n=== Trying with manual FreeSurfer data ===")
        # Use the provided FreeSurfer statistics as sample data
        results = predictor.run_full_pipeline(
            feature_type='volumetric',
            use_feature_selection=True,
            use_manual_data=True
        )
        
        if results:
            best_model = min(results.keys(), key=lambda x: results[x]['cv_mae'])
            print(f"\n=== Final Results (Manual Data) ===")
            print(f"Best Model: {best_model}")
            print(f"Cross-validation MAE: {results[best_model]['cv_mae']:.2f} years")
            print(f"Test MAE: {results[best_model]['test_mae']:.2f} years")
            print(f"Test R²: {results[best_model]['test_r2']:.3f}")
        else:
            print("Failed to run pipeline with manual data")