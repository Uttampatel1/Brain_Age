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
                    else:
                        volumes = pd.read_csv(volume_file)
                    
                    # Create feature dictionary
                    feature_dict = {
                        'subject_id': subject_id,
                        'age': age,
                        'gender': 1 if gender == 'M' else 0,
                        'folder_name': folder
                    }
                    
                    # Add volumetric features
                    for idx, row in volumes.iterrows():
                        if len(volumes.columns) >= 2:
                            region_name = str(row.iloc[0]).replace(' ', '_').replace('-', '_')
                            volume_value = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0
                            feature_dict[f'vol_{region_name}'] = volume_value
                    
                    data_list.append(feature_dict)
                    
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
            
        selector = SelectKBest(score_func=f_regression, k=min(k, self.features.shape[1]))
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42)
        }
        
        results = {}
        
        print("Training models...")
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=5, scoring='neg_mean_absolute_error')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
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
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Define parameter grids based on model type
        if isinstance(self.best_model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif isinstance(self.best_model, GradientBoostingRegressor):
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif isinstance(self.best_model, SVR):
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        else:
            print("No hyperparameter optimization available for this model")
            return
        
        print("Optimizing hyperparameters...")
        grid_search = GridSearchCV(
            self.best_model, param_grid, cv=5, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        self.best_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: {-grid_search.best_score_:.2f}")
    
    def plot_results(self, results):
        """Plot model performance and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_names = list(results.keys())
        cv_maes = [results[name]['cv_mae'] for name in model_names]
        test_maes = [results[name]['test_mae'] for name in model_names]
        
        axes[0, 0].bar(range(len(model_names)), cv_maes, alpha=0.7, label='CV MAE')
        axes[0, 0].bar(range(len(model_names)), test_maes, alpha=0.7, label='Test MAE')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        
        # Best model predictions vs actual
        best_name = min(results.keys(), key=lambda x: results[x]['cv_mae'])
        y_pred = results[best_name]['predictions']
        y_actual = results[best_name]['actual']
        
        axes[0, 1].scatter(y_actual, y_pred, alpha=0.6)
        axes[0, 1].plot([y_actual.min(), y_actual.max()], 
                       [y_actual.min(), y_actual.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Age')
        axes[0, 1].set_ylabel('Predicted Age')
        axes[0, 1].set_title(f'Best Model ({best_name}) Predictions')
        
        # Residuals plot
        residuals = y_actual - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Age')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Plot')
        
        # Age distribution
        axes[1, 1].hist(self.targets, bins=20, alpha=0.7, label='Actual Ages')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Age Distribution in Dataset')
        axes[1, 1].legend()
        
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
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    
    def predict_age(self, new_features):
        """Predict age for new data"""
        if self.best_model is None:
            print("No model trained!")
            return None
        
        new_features_scaled = self.scaler.transform(new_features)
        prediction = self.best_model.predict(new_features_scaled)
        return prediction
    
    def run_full_pipeline(self, feature_type='volumetric', use_feature_selection=True):
        """Run the complete brain age prediction pipeline"""
        print("Starting Brain Age Prediction Pipeline...")
        
        # Load data
        if feature_type == 'volumetric':
            success = self.load_volumetric_features()
        else:
            success = self.load_image_features()
        
        if not success:
            print("Failed to load data!")
            return None
        
        # Prepare features
        self.prepare_features()
        
        # Feature selection
        if use_feature_selection and self.features.shape[1] > 50:
            self.feature_selection(k=min(50, self.features.shape[1] // 2))
        
        # Train models
        results = self.train_models()
        
        # Optimize best model
        self.optimize_best_model()
        
        # Plot results
        self.plot_results(results)
        
        return results

# Usage example
if __name__ == "__main__":
    # Set your data path
    data_path = r"A:\Uttam\Apps\Brain_Age\output"  # Update this path
    
    # Create predictor instance
    predictor = BrainAgePredictor(data_path)
    
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