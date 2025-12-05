基于 SVM（支持向量机） 实现的 emotion classification model

代码说明：
各 .py 文件含有各部分使用到的函数的定义，其实现在 Pipeline_SVM.ipynb 文件

load_data.py :
	含有读取数据集的代码

	read_json(file):
		读取训练集，即 *_train.txt

training_SVM.py :
	构建SVM模型

	build_vectorizer():
		构建 TfidfVectorizer

	tune_linear_svm_with_cv(texts, label, search_mode, cv, n_iter):
		通过 build_vectorizer 向量化训练数据，并构建 LinearSVC 模型。
		其次，使用 RandomizedSearchCV 查找模型最佳的 C值

evaluation.py :
	模型评估使用到的函数

	load_eval_pair(eval_path, labeled_path):
		读取评价（evaluation）数据集，即 *_eval.txt 与 *_eval_labeled.txt
		并输出 *_eval_labeled 数据集的 ids, texts, true_labels（正确的情感标签）
	
	evaluate_eval_files(model, vectorizer, eval_path, labeled_path):
		对 *_eval 数据集，使用SVM模型进行测试，并于 true_labels 进行比较。
		其效果评估实现使用了 classification_report、accuracy_score、confusion_matrix

Pipeline_SVM.ipynb :
	使用已定义的函数，实现训练以及评估。
	模型训练结果存储到 .pkl
	测试时还使用到了 test_virus 和 test_virus_label 数据，
	测试方法与评估方法相同，直接使用了 evaluate_eval_files

Self_Test.ipynb : 
	使用模型训练结果，可以自主输入中文句子，并查看基于模型分类的情感标签结果。
