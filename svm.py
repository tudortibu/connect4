import utility as util
from sklearn import svm
def main(train_path,  test_path):

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')
    train_x_inter = util.add_intercept(train_x)
    test_x_inter = util.add_intercept(test_x)
    svc = svm.LinearSVC()

    svc.fit(train_x, train_t)
    pred_svc_prob = svc.predict(test_x)

    for i in range(len(test_t)):
        print(f"Predicted svm value: {pred_svc_prob[i]} True Value: {test_t[i]}")

main("Connect4_Data/8-ply Moves/connect-4-clean-train.csv", "Connect4_Data/8-ply Moves/connect-4-clean-test.csv")
