import pandas as pd
import math
import numpy as np
from collections import OrderedDict
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import math
from imblearn.over_sampling import SMOTE
from copy import deepcopy
from collections import deque

class Node():
    def __init__(self, feature=None, threshold=None, children=None , gain=None, value=None, feature_mapping=None, majority_value=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.gain = gain
        self.value = value
        self.feature_mapping = feature_mapping  
        self.majority_value = majority_value
    def print_contents(self):
        print('---------------')
        print('feature',self.feature)
        print('threshold',self.threshold)
        print('children',self.children)
        print('gain',self.gain)
        print('value',self.value)
        print('feature mapping',self.feature_mapping)
        print('---------------')


class DecisionTree():
    def __init__(self, min_samples=2, max_depth=5):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root=Node()
  
    def calculate_entropy(self, data):
        total_count = len(data)
        if total_count == 0:
            return 0
        class_counts = data['y'].value_counts()
        entropy = 0
        for count in class_counts:
            if (count==0 or count==total_count):
              continue
            else:
              probability = count / total_count
              entropy -= probability * math.log2(probability)
        return entropy
    
    def calculate_gini_impurity(self, dataset, feature):
        gini_impurity = 0
        total_count = len(dataset)

        if total_count == 0:
          return 0

        unique_values = dataset[feature].unique()
        for value in unique_values:
          subset = dataset[dataset[feature] == value]
          subset_count = len(subset)
          if subset_count > 0:
              class_counts = subset['y'].value_counts()
              subset_gini = 1
              for count in class_counts:
                probability = count / subset_count
                subset_gini -= probability ** 2

              gini_impurity += (subset_count / total_count) * subset_gini

        return gini_impurity
    
    def calculate_information_gain(self, data, feature):
        original_entropy = self.calculate_entropy(data)
        feature_values = data[feature].unique()
        weighted_entropy = 0

        for value in feature_values:
            subset = data[data[feature] == value]
            subset_entropy = self.calculate_entropy(subset)
            weight = len(subset) / len(data)
            weighted_entropy += weight * subset_entropy

        information_gain = original_entropy - weighted_entropy
        return information_gain


    def find_best_split(self, data, split='ig'):
        if (split=='ig'):
            best_feature = None
            best_gain = -1
            for feature in data.columns[:-1]:  # Exclude the target variable column
                gain = self.calculate_information_gain(data, feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
            return best_feature, best_gain
        else:
            best_feature = None
            best_gain = 2
            for feature in data.columns[:-1]:  # Exclude the target variable column
                gain = self.calculate_gini_impurity(data, feature)
                if gain < best_gain:
                    best_gain = gain
                    best_feature = feature
            return best_feature, best_gain


    def split_data_categorical(self, dataset, feature):
        value_map = OrderedDict()
        if feature is not None:
            unique_values = sorted(dataset[feature].unique())
            for value in unique_values:
                subset = dataset[dataset[feature] == value].reset_index(drop=True)
                value_map[value] = subset
        return value_map

    def calculate_leaf_value(self,y):
        most_occuring_value = y.value_counts().idxmax()
        return most_occuring_value
    
    def build_tree(self, dataset, current_depth):
        n_samples = dataset.shape[0]
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            best_split_feature, best_information_gain = self.find_best_split(dataset)
            if best_split_feature!=None:
              node_children=[]
              datasets=self.split_data_categorical(dataset, best_split_feature)
              for key in datasets:
                  node_children.append(self.build_tree(datasets[key], current_depth+1))

              feature_map=OrderedDict()
              unique_values = sorted(dataset[best_split_feature].unique())

              for i, value in enumerate(unique_values):
                    feature_map[value] = i
              majority_value=dataset['y'].value_counts().idxmax()
              return Node(best_split_feature, None, node_children, best_information_gain, None, feature_map, majority_value)
            else:
              return Node()
        else:
            leaf_value = self.calculate_leaf_value(dataset['y'])
            return Node(value=leaf_value)

    def cost_complexity_prune(self, node, validation_data):
        if node.children is None:
            return node
        pruned_children = []
        for child in node.children:
            pruned_children.append(self.cost_complexity_prune(child, validation_data))
        node.children = pruned_children
        original_accuracy = self.evaluate_accuracy(validation_data)
        leaf_value = node.majority_value
        pruned_node = Node(value=leaf_value)
        self.replace_node(node, pruned_node)
        pruned_accuracy = self.evaluate_accuracy(validation_data)

        if pruned_accuracy < original_accuracy:
            self.replace_node(pruned_node, node)
            return node
        else:
            return pruned_node

    def replace_node(self, old_node, new_node):
        if self.root == old_node:
            self.root = new_node
        else:
            queue = deque([self.root])  
            while queue:
                current = queue.popleft()  
                if current.children is not None:
                    for i in range(len(current.children)):
                        if current.children[i] == old_node:
                            current.children[i] = new_node
                            return  
                        queue.append(current.children[i])

    def evaluate_accuracy(self, validation_data):
        X_val = validation_data.drop(columns=['y'])
        y_val = validation_data['y']
        y_pred = self.predict(X_val)
        return self.accuracy(y_val, y_pred)

    def prune_tree(self, X_val, y_val):
        validation_data = X_val.copy()
        validation_data['y'] = y_val
        self.root = self.cost_complexity_prune(self.root, validation_data)

    def fit(self, X, y):
        dataset = X.copy()
        dataset['y'] = y
        self.root = self.build_tree(dataset, 0)
        return self
    
    def make_prediction(self, x, node):
        if node.value != None and node.children==None: 
            return node.value
        else:
            feature = x[node.feature]
            if feature not in node.feature_mapping:
                return node.majority_value
            index=node.feature_mapping[feature]
            return self.make_prediction(x, node.children[index])

    def predict(self, X):
        predictions = []
        for _, x in X.iterrows():
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)
        np.array(predictions)
        return predictions

    def accuracy(self, y_true, y_pred):
        correct_predictions = (y_true == y_pred).sum()
        accuracy = correct_predictions / len(y_true)

        return accuracy
    
    def precision(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)  # Ensure consistency

        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)

        if predicted_positives == 0:
            return 0.0  # Avoid division by zero

        return true_positives / predicted_positives

    def recall(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)  # Ensure consistency

        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)

        if actual_positives == 0:
            return 0.0  # Avoid division by zero

        return true_positives / actual_positives
    
    def f1_score(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        if precision + recall == 0:
            return 0.0  # Avoid division by zero

        return 2 * (precision * recall) / (precision + recall)
    
    
def depth_of_tree(node):
    if node.value!=None:
        return 0
    depth_children=0
    for child in node.children:
        depth_children=max(depth_children,depth_of_tree(child))
    return 1+depth_children

def number_of_nodes(node):
    if node.value!=None:
        return 1
    children_num=0
    for child in node.children:
        children_num+=number_of_nodes(child)
    return 1+children_num

dataset = pd.read_excel("Train.xlsx")
train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=234)

if train_df['y'].dtype == 'object':
    train_df['y'] = train_df['y'].map({'no': 0, 'yes': 1})
if 'y' in val_df.columns and val_df['y'].dtype == 'object':
    val_df['y'] = val_df['y'].map({'no': 0, 'yes': 1})


numerical_columns = [col for col in dataset.columns if pd.api.types.is_numeric_dtype(dataset[col]) and col != 'y']
categorical_columns = [col for col in dataset.columns if not pd.api.types.is_numeric_dtype(dataset[col]) and col != 'y']

def apply_kmeans_binning(train_data, val_data, num_bins=5):
    binned_feature_names = []  
    kmeans_models = {}        
    for feature in numerical_columns:
        try:
            train_values = train_data[feature].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_bins, random_state=567)
            kmeans.fit(train_values)
            kmeans_models[feature] = kmeans
            train_data[feature + '_bin'] = kmeans.predict(train_values)
            val_values = val_data[feature].values.reshape(-1, 1)
            val_data[feature + '_bin'] = kmeans.predict(val_values)
            for bin_num in range(num_bins):
                new_column_name = f"{feature}_bin_{bin_num}"
                binned_feature_names.append(new_column_name)
                train_data[new_column_name] = (train_data[feature + '_bin'] == bin_num).astype(int)
                val_data[new_column_name] = (val_data[feature + '_bin'] == bin_num).astype(int)

        except ValueError as e:
            print(f"Could not create bins for column '{feature}': {e}")

    train_data.drop(columns=[feature + '_bin' for feature in numerical_columns if feature + '_bin' in train_data.columns], inplace=True)
    val_data.drop(columns=[feature + '_bin' for feature in numerical_columns if feature + '_bin' in val_data.columns], inplace=True)

    return train_data, val_data, binned_feature_names, kmeans_models

train_df, val_df, created_bin_features, kmeans_models = apply_kmeans_binning(train_df, val_df)
final_train_columns = created_bin_features + categorical_columns + ['y']
final_val_columns = created_bin_features + categorical_columns

final_train_df = train_df[final_train_columns]
final_val_df = val_df[final_val_columns + ['y']]


non_numeric_columns_train = final_train_df.select_dtypes(exclude=['number']).columns
non_numeric_columns_val = final_val_df.select_dtypes(exclude=['number']).columns

final_train_df_encoded = pd.get_dummies(final_train_df, columns=non_numeric_columns_train, drop_first=True)
final_val_df_encoded = pd.get_dummies(final_val_df, columns=non_numeric_columns_val, drop_first=True)

final_train_df_encoded, final_val_df_encoded = final_train_df_encoded.align(final_val_df_encoded, join='left', axis=1, fill_value=0)




# print(final_train_df_encoded.loc[18126])

# def map_samples_to_features(data, created_bin_features):
#     """Maps samples to features based on one-hot encoded bin values.
    
#     Args:
#         data (pd.DataFrame): The dataframe with one-hot encoded bin features.
#         created_bin_features (list): List of one-hot encoded feature names.

#     Returns:
#         pd.DataFrame: A dataframe that maps each sample to the activated features.
#     """
#     # Create a mapping dataframe to record active features
#     feature_map = pd.DataFrame(index=data.index)

#     # Iterate through each sample and each one-hot encoded feature
#     for sample_index in data.index:
#         active_features = [feature for feature in created_bin_features if data.loc[sample_index, feature] == 1]
#         feature_map.loc[sample_index, 'active_features'] = ', '.join(active_features) if active_features else 'None'
    
#     return feature_map


# # Use the mapping function on your train and validation dataframes
# train_feature_map = map_samples_to_features(final_train_df, created_bin_features)
# val_feature_map = map_samples_to_features(final_val_df, created_bin_features)

# # Print or inspect the first few entries
# print("Train Sample Feature Map:")
# print(train_feature_map.head())

# print("\nValidation Sample Feature Map:")
# print(val_feature_map.head())

# Print the list of newly created bin-based features
# print(created_bin_features)


dt = DecisionTree(max_depth=10, min_samples=3)

# # Fit the model using the final transformed training data
# X_train = final_train_df.drop(columns='y')  # Features for training
# y_train = final_train_df['y']                # Target for training

X_train = final_train_df_encoded.drop(columns='y')  # Features for training
y_train = final_train_df_encoded['y']

# dt.fit(X_train, y_train)

# smote = SMOTE()
X_train=final_train_df_encoded.drop(columns='y')
y_train=final_train_df_encoded['y']
# print(X_train.shape)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# print(X_resampled.shape)
# dt.fit(X_resampled, y_resampled)
dt.fit(X_train, y_train)
# print(dt.root.children[1].print_contents())
# print(dt.root.print_contents())
# # Make predictions on the validation data
# X_val = final_val_df.drop(columns='y')  # Features for validation
# y_val = final_val_df['y']                # Target for validation
X_val = final_val_df_encoded.drop(columns='y')  # Features for validation
y_val = final_val_df_encoded['y']


y_pred=dt.predict(X_val)
accuracy = dt.accuracy(y_val, y_pred)
recall=dt.recall(y_val,y_pred)
precision=dt.precision(y_val,y_pred)
f1_score=dt.f1_score(y_val,y_pred)

# Print the validation accuracy
print(f"Validation Accuracy: {accuracy:.8f}")
print(f"recall: {recall:.8f}")
print(f"Precision: {precision:.8f}")
print(f"F1 score: {f1_score:.8f}")

print(depth_of_tree(dt.root))
print(number_of_nodes(dt.root))

dt.prune_tree(X_val, y_val)

y_pred_pruned = dt.predict(X_val)
pruned_accuracy = dt.accuracy(y_val, y_pred_pruned)
pruned_recall=dt.recall(y_val, y_pred_pruned)
pruned_precision=dt.precision(y_val, y_pred_pruned)
pruned_f1_score=dt.f1_score(y_val,y_pred_pruned)
print(f"Accuracy after pruning: {pruned_accuracy:.8f}")
print(f"Recall: {pruned_recall:.8f}")
print(f"Precision: {pruned_precision:.8f}")
print(f"F1 score: {pruned_f1_score:.8f}")
dt.root.print_contents()
print(depth_of_tree(dt.root))
print(number_of_nodes(dt.root))



