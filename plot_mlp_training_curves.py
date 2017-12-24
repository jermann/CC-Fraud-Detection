"""
========================================================
Compare Stochastic learning strategies for MLPClassifier
========================================================

This example visualizes some training loss curves for different stochastic
learning strategies, including SGD and Adam. Because of time-constraints, we
use several small datasets, for which L-BFGS might be more suitable. The
general trend shown in these examples seems to carry over to larger datasets,
however.

Note that those results can be highly dependent on the value of
``learning_rate_init``.
"""

print(__doc__)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01},
          {'hidden_layer_sizes': (18,12,8), 'activation': 'logistic', 'solver':'lbfgs'}
         ]

labels = ["Constant Learning-Rate", "Constant with Momentum","Constant with Nesterov's Momentum","Inv-Scaling with Momentum", "Adam","Original"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '--'}
            ]


def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 200

    for label, param in zip(labels, params):
        print("Parameters: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0, max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %.4f" % mlp.score(X, y))
        print("Training set loss:  %.4f \n" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
        if label != 'Original':
            ax.plot(mlp.loss_curve_, label=label, **args)
        else:
            break



fig, axes = plt.subplots(figsize=(15, 10))

plot_on_dataset(X_train_un, y_train_un_1d, ax=axes, name='credit')

fig.legend(ax.get_lines(), labels=labels, ncol=3, loc="upper center")
plt.show()
