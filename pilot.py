import os
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from tabulate import tabulate
from colorama import init, Fore, Style

# 初始化colorama
init(autoreset=True)

# 下载停用词列表
nltk.download('stopwords')

# 数据加载函数
def load_data(data_dir, num_samples):
    data = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        files = os.listdir(dir_name)[:num_samples]
        for fname in files:
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                    data.append(f.read())
                    labels.append(1 if label_type == 'pos' else 0)
    return data, labels

# 加载训练集和测试集（各1000条正面评论和1000条负面评论）
train_data, train_labels = load_data('aclImdb/train', 1000)
test_data, test_labels = load_data('aclImdb/test', 1000)

# 文本预处理函数
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text) # 移除HTML标签
    text = re.sub(r'[^\w\s]', '', text) # 移除标点符号和特殊字符
    text = text.lower() # 转换为小写
    return text

# 预处理数据
train_data = [preprocess_text(review) for review in train_data]
test_data = [preprocess_text(review) for review in test_data]

# 移除停用词
stop_words = set(stopwords.words('english'))
train_data = [' '.join([word for word in review.split() if word not in stop_words]) for review in train_data]
test_data = [' '.join([word for word in review.split() if word not in stop_words]) for review in test_data]

# 特征提取
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# 模型定义
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Bagging": BaggingClassifier(n_estimators=100)
}

# 模型训练和评估
results = []
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, train_labels)
    # 预测
    predictions = model.predict(X_test)
    # 评估模型
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 打印结果表格
print(Fore.BLUE + Style.BRIGHT + "Model Performance Results:")
print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False, colalign=("center", "center", "center", "center", "center")))
