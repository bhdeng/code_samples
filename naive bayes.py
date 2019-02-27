import numpy as np
import pandas
import os
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# index for continuous features
continuous_feature_index = [0,2,4,10,11,12]
# features
features = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"]
# discreate features map to possible values
feature_dicts = {"workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", 
                               "Federal-gov", "Local-gov", "State-gov", "Without-pay", 
                               "Never-worked"],
                 "education": ["Bachelors", "Some-college", "11th", "HS-grad", 
                               "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", 
                               "12th", "Masters", "1st-4th", "10th", "Doctorate", 
                               "5th-6th", "Preschool"],
                 "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", 
                                    "Separated", "Widowed", "Married-spouse-absent", 
                                    "Married-AF-spouse"],
                 "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                                "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                                "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                                "Transport-moving", "Priv-house-serv", "Protective-serv", 
                                "Armed-Forces"],
                 "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", 
                                  "Other-relative", "Unmarried"],
                 "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", 
                          "Black"],
                 "sex": ["Female", "Male"],
                 "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", 
                                    "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", 
                                    "India", "Japan", "Greece", "South", "China", "Cuba", 
                                    "Iran", "Honduras", "Philippines", "Italy", "Poland", 
                                    "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", 
                                    "France", "Dominican-Republic", "Laos", "Ecuador", 
                                    "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", 
                                    "Nicaragua", "Scotland", "Thailand", "Yugoslavia", 
                                    "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", 
                                    "Holand-Netherlands"]
                }

# read and preprocess data (ignore the data entries with ?)
def read_and_preprocess(filename, mode="train", num_samples=None):
    if mode == "train":
        df = pandas.read_csv(os.path.join(DATA_PATH, filename), 
                             skipinitialspace=True, header=-1)
    else:
        df = pandas.read_csv(os.path.join(DATA_PATH, filename), 
                             skipinitialspace=True, skiprows=1, header=-1)
    if num_samples:
        df = df[:num_samples]
    # ignore the unknown data entries
    val_idx = df.index[~(df=='?').any(axis=1)]
    df = df[~(df=='?').any(axis=1)]

    return val_idx, df

# count for computing parameters of discrete features
def count_dist(df, feature_idx):
    value_count = df.iloc[:,feature_idx].value_counts()
    value = value_count.keys().tolist()
    count = value_count.tolist()
    return value, count

# compute the class prior
def class_prior(df, if_print=False):
    N = len(df.index)
    label, count = count_dist(df, -1)
    prior = [c / N for c in count]

    if if_print:
        for l, p in zip(label, prior):
            print ("prior for class %s: %.4f" % (l, p))

    return label, prior

# compute the parameters for each feature
def train(df, if_print=False):
    label_value, label_count = count_dist(df, -1)
    param_dicts = {}
    
    for label in label_value:
        param_dict_per_label = {}
        label_data = df[df.iloc[:,-1] == label]
        for idx in range(label_data.shape[1]-1):
            if idx not in continuous_feature_index:
                value, count = count_dist(label_data, idx)
                param = [c / len(label_data.index) for c in count]
                temp_param_dict = dict(zip(value, param))

                zero_param_values = list(set(feature_dicts[features[idx]]) - set(value))
                temp_param_dict.update({k: 0.0 for k in zero_param_values})

                param_dict_per_label[features[idx]] = temp_param_dict
            else:
                mean = np.mean(label_data[idx])
                var = np.var(label_data[idx])
                param_dict_per_label[features[idx]] = {"mean": mean, "variance": var}

        param_dicts[label] = param_dict_per_label

    if if_print:
        for class_label in param_dicts:
            print ("class %s:" % class_label)
            for feature in features:
                print ("\t%s: " % feature, end="")
                if feature in feature_dicts:
                    for value in feature_dicts[feature]:
                        print ("%s=%.4f" % (value, param_dicts[class_label][feature][value]), 
                               end=", ")
                else:
                    for value in ["mean", "variance"]:
                        print ("%s=%.4f" % (value, param_dicts[class_label][feature][value]), 
                               end=", ")
                print ("\n", end="")
            print ("\n", end="")

    return param_dicts

# evaluate the model with trained parameters
def test(val_idx, df, label, prior, param_dicts, if_print=False):
    epsilon = 1e-9
    num_samples = len(df.index)
    log_posterior_per_sample = []

    # iterate through each sample
    for sample in val_idx:
        log_posterior = [np.log(prior[0]), np.log(prior[1])]

        for i in range(len(log_posterior)):
            for f_idx, feature in enumerate(features):
                value = df[f_idx][sample]
                if f_idx not in continuous_feature_index:
                    prob = param_dicts[label[i]][feature][value]
                    log_posterior[i] += np.log(prob)
                else:
                    mean = param_dicts[label[i]][feature]["mean"]
                    var = param_dicts[label[i]][feature]["variance"] + epsilon
                    log_posterior[i] -= np.log(np.sqrt(2*np.pi*var)) + (value - mean)**2 / (2*var)

        log_posterior_per_sample.append(log_posterior)

    if if_print:
        for i, p in zip(val_idx, log_posterior_per_sample):
            print ("For data at line %d, log-posterior for class %s is %.4f; for class %s is %.4f"
                   % (i+2, label[0], p[0], label[1], p[1]))

    pred_labels = [label[0] if p[0] > p[1] else label[1] for p in log_posterior_per_sample]
    true_labels = df.iloc[:,-1].str.replace('.', '').tolist()
    accuracy = sum(1 for pred_l, true_l in zip(pred_labels, true_labels) 
                   if pred_l == true_l) / len(true_labels)

    return accuracy

# Q5.1a
def question_51a():
    _, data = read_and_preprocess("adult.data")
    class_prior(data, True)

# Q5.1b
def question_51b():
    _, data = read_and_preprocess("adult.data")
    train(data, True)

# Q5.1c
def question_51c():
    _, data = read_and_preprocess("adult.data")
    val_test_idx, test_data = read_and_preprocess("adult.test", mode="test", num_samples=10)

    label, prior = class_prior(data)
    param_dicts = train(data)
    test(val_test_idx, test_data, label, prior, param_dicts, if_print=True)

# Q5.2a
def question_52a():
    val_train_idx, data = read_and_preprocess("adult.data")
    label, prior = class_prior(data)
    param_dicts = train(data)

    train_accuracy = test(val_train_idx, data, label, prior, param_dicts)
    print ("training accuracy: %.4f" % train_accuracy)

# Q5.2b
def question_52b():
    _, data = read_and_preprocess("adult.data")
    val_test_idx, test_data = read_and_preprocess("adult.test", mode="test")

    label, prior = class_prior(data)
    param_dicts = train(data)
    test_accuracy = test(val_test_idx, test_data, label, prior, param_dicts)
    print ("testing accuracy: %.4f" % test_accuracy)

# Q5.2c
def question_52c():
    val_test_idx, test_data = read_and_preprocess("adult.test", mode="test")
    data = pandas.read_csv(os.path.join(DATA_PATH, "adult.data"), 
                           skipinitialspace=True, header=-1)

    num_samples = [2**i for i in range(5,14)]
    train_accuracy = []
    test_accuracy = []
    for n in num_samples:
        train_data = data[:n]
        val_train_idx = train_data.index[~(train_data=='?').any(axis=1)]
        train_data = train_data[~(train_data=='?').any(axis=1)]

        label, prior = class_prior(train_data)
        param_dicts = train(train_data)
        train_accuracy.append(test(val_train_idx, train_data, label, prior, param_dicts))
        test_accuracy.append(test(val_test_idx, test_data, label, prior, param_dicts))

    for n, accuracy in zip(num_samples, test_accuracy):
        print ("testing accuracy for n = %d is %.4f" % (n, accuracy))

    print ("maximum training accuracy at n = %d" % num_samples[np.argmax(train_accuracy)])
    print ("maximum testing accuracy at n = %d" % num_samples[np.argmax(test_accuracy)])
    
    plt.xlabel("number of samples")
    plt.ylabel("accuracy")
    plt.plot(num_samples, train_accuracy, label="training accuracy")
    plt.plot(num_samples, test_accuracy, label="testing accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #question_51a()
    #question_51b()
    #question_51c()
    #question_52a()
    #question_52b()
    question_52c()
    
