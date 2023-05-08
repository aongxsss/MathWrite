# Load Data , Data Processing
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ROCAUC
import pickle
load_dotenv()
path = os.getenv('IMAGE_PATH')
print(path)
symbol_classes = ['+', '-', 'x','÷', ')', '(']
num_images = 1250
data = []
labels = []
for symbol in symbol_classes:
    symbol_path = os.path.join(path, symbol)
#     print(symbol_path)
#     print(os.listdir(symbol_path))
    image_files = os.listdir(symbol_path)
    np.random.shuffle(image_files)
    
    image_files = image_files[:num_images]
    
    for image_file in image_files:
        image_path = os.path.join(symbol_path, image_file)
        image = Image.open(image_path).resize((28, 28))
        image = image.convert('L')
        image_array = np.array(image) / 255.0
        data.append(image_array)
        labels.append(symbol_classes.index(symbol))
        
data = np.array(data)
labels = np.array(labels)

print('Data shape:', data.shape)
print('Labels shape:', labels.shape)

# Split Data , Train Model with Decision Tree Algorithm
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train_reshaped  = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

model = DecisionTreeClassifier(max_depth=16)
model.fit(X_train_reshaped, y_train)

y_pred = model.predict(X_test_reshaped)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test_reshaped)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# plot confusion matrix as heatmap with custom labels
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=symbol_classes, yticklabels=symbol_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=symbol_classes))

# ROC&AUC plot
visualizer = ROCAUC(model, classes=symbol_classes)
visualizer.fit(X_test_reshaped, y_train)
visualizer.score(X_test_reshaped , y_test)
visualizer.show()


#Save Model
with open('best.pkl', 'wb') as f:
    pickle.dump(model, f)