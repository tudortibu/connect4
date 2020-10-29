import utility as util
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree as t
import graphviz
import matplotlib.pyplot as plt

def main(train_path,  test_path):

    train_x, train_t = util.load_dataset(train_path, label_col='class')
    test_x, test_t = util.load_dataset(test_path, label_col='class')

    tree = DecisionTreeClassifier()
    tree.fit(train_x, train_t)
    #fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,15), dpi=500)
    #t.plot_tree(tree, filled=True)
    #fig.savefig('plot.png')
    pred_tree_prob = tree.predict(test_x)

    for i in range(len(test_t)):
        print(f"Predicted Decision Tree value: {pred_tree_prob[i]} True Value: {test_t[i]}")

main("Connect4_Data/8-ply Moves/connect-4-clean-train.csv", "Connect4_Data/8-ply Moves/connect-4-clean-test.csv")