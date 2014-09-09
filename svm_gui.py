# -*- coding: utf-8 -*-
"""
==========
Libsvm GUI
==========

A simple graphical frontend for Libsvm mainly intended for didactic
purposes. You can create data points by point and click and visualize
the decision region induced by different kernels and parameter settings.

To create positive examples click the left mouse button; to create
negative examples click the right button.

If all examples are from the same class, it uses a one-class SVM.

Extension: this extension allows to use epsilon-SVM for regression (SVR).
Instead of plotting decision surfaces the epsilon tube and support vectors
are drawn.

"""
from __future__ import division, print_function

print(__doc__)

# Author: Peter Prettenhoer <peter.prettenhofer@gmail.com> (original svm_gui example)
#         Javier Sánchez <jsanchezm@uco.es> (extension to regression case)
#
# License: BSD 3 clause

import matplotlib

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet

import Tkinter as Tk
import tkFileDialog
import sys
import numpy as np

from sklearn import svm
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.externals.six.moves import xrange

y_min, y_max = -50, 50
x_min, x_max = -50, 50


class Model(object):
    """The Model which hold the data. It implements the
    observable in the observer pattern and notifies the
    registered observers on change event.
    """

    def __init__(self):
        self.observers = []
        self.surface = None
        self.data = []
        self.cls = None
        self.surface_type = 0

    def changed(self, event, **kwargs):
        """Notify the observers. """
        for observer in self.observers:
            observer.update(event, self, kwargs)

    def add_observer(self, observer):
        """Register an observer. """
        self.observers.append(observer)

    def set_surface(self, surface):
        self.surface = surface

    def load_svmlight_file(self, file):
        X, y = load_svmlight_file(file)
        self.data = []
        self.data =  np.concatenate((X.toarray(), np.transpose([y])), axis=1).tolist()

    def dump_svmlight_file(self, file):
        data = np.array(self.data)
        X = data[:, 0:2]
        y = data[:, 2]
        dump_svmlight_file(X, y, file)

class Controller(object):
    def __init__(self, model):
        self.model = model
        self.kernel = Tk.IntVar()
        self.surface_type = Tk.IntVar()
        self.classification = Tk.IntVar()
        # Whether or not a model has been fitted
        self.fitted = False

    def fit(self):
        train = np.array(self.model.data)

        C = float(self.complexity.get())
        gamma = float(self.gamma.get())
        epsilon = float(self.epsilon.get())
        coef0 = float(self.coef0.get())
        degree = int(self.degree.get())
        # epsilon = float(self.epsilon.get())
        kernel_map = {0: "linear", 1: "rbf", 2: "poly"}

        if self.classification.get() == 1:

            X = train[:, 0:2]
            y = train[:, 2]
            if len(np.unique(y)) == 1:
                clf = svm.OneClassSVM(kernel=kernel_map[self.kernel.get()],
                                      gamma=gamma, coef0=coef0, degree=degree)
                clf.fit(X)
            else:
                clf = svm.SVC(kernel=kernel_map[self.kernel.get()], C=C,
                              gamma=gamma, coef0=coef0, degree=degree)
                clf.fit(X, y)
            if hasattr(clf, 'score'):
                print("Accuracy:", clf.score(X, y) * 100)
            X1, X2, Z = self.decision_surface(clf)
            self.model.clf = clf
            self.model.set_surface((X1, X2, Z))
            self.model.surface_type = self.surface_type.get()
            self.fitted = True
            self.model.changed("surface")

        else:
            X = train[:, 0:1]
            y = train[:, 1:2]
            y=y[:,0]

            clf = svm.SVR(kernel=kernel_map[self.kernel.get()], C=C,
                          gamma=gamma, coef0=coef0, degree=degree, epsilon=epsilon)
            y_pred = clf.fit(X, y).predict(X)

            self.model.clf = clf
            # TODO: These data does not belong to the model, so maybe they should be placed outside
            self.model.y_pred = y_pred
            self.model.X = X
            self.model.y = y
            self.fitted = True

            self.model.changed("tube", X=X, y=y, y_pred=y_pred)

    def decision_surface(self, cls):
        delta = 1
        x = np.arange(x_min, x_max + delta, delta)
        y = np.arange(y_min, y_max + delta, delta)
        X1, X2 = np.meshgrid(x, y)
        Z = cls.decision_function(np.c_[X1.ravel(), X2.ravel()])
        Z = Z.reshape(X1.shape)
        return X1, X2, Z

    def clear_data(self):
        self.model.data = []
        self.fitted = False
        self.model.changed("clear")

    def add_example(self, x, y, label):
        self.model.data.append((x, y, label))
        self.model.changed("example_added")

        # update decision surface if already fitted.
        self.refit()

    def refit(self):
        """Refit the model if already fitted. """
        if self.fitted:
            self.fit()

    def open_file(self, file, format='csv'):
        self.model.changed("clear")
        self.model.load_svmlight_file(file)
        self.model.changed("examples_loaded")


    def save_file(self, file, format='csv'):
        self.model.dump_svmlight_file(file)


class View(object):
    """Test docstring. """

    def __init__(self, root, controller):
        f = Figure()
        ax = f.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas.mpl_connect('button_press_event', self.onclick)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        self.controllbar = ControllBar(root, controller)
        self.menubar = MenuBar(root, controller)
        self.f = f
        self.ax = ax
        self.canvas = canvas
        self.controller = controller
        self.contours = []
        self.tube = []
        self.c_labels = None
        self.plot_kernels()

        self.class_colors = ('b', 'r', 'g', 'y')

    def plot_kernels(self):
        self.ax.text(-50, -60, "Linear: $u^T v$")
        self.ax.text(-20, -60, "RBF: $\exp (-\gamma \| u-v \|^2)$")
        self.ax.text(10, -60, "Poly: $(\gamma \, u^T v + r)^d$")

    def onclick(self, event):
        # TODO: Update for multiclass
        if event.xdata and event.ydata:
            if event.button == 1:
                self.controller.add_example(event.xdata, event.ydata, 1)
            elif event.button == 3:
                self.controller.add_example(event.xdata, event.ydata, -1 if self.controller.classification.get() else 1)


    def update_example(self, model, idx):
        x, y, l = model.data[idx]
        if l == 1:
            color = 'w'
        elif l == -1:
            color = 'k'
        self.ax.plot([x], [y], "%so" % color, scalex=0.0, scaley=0.0)

    def update(self, event, model, kwargs):
        # args for passing y_pred needed for plotting the SVR model
        if event == "examples_loaded":
            for i in xrange(len(model.data)):
                self.update_example(model, i)

        if event == "example_added":
            self.update_example(model, -1)

        if event == "clear":
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.contours = []
            self.tube = []
            self.c_labels = None
            self.plot_kernels()

        if event == "surface":
            self.remove_surface()
            self.remove_tube()
            self.plot_support_vectors(model.clf.support_vectors_)
            self.plot_decision_surface(model.surface, model.surface_type)

        if event == "tube":
            self.remove_surface()
            self.remove_tube()
            self.plot_support_vectors_svr(model.clf.support_, X=kwargs['X'], y=kwargs['y'])
            self.plot_epsilon_tube(model.clf.epsilon, X=kwargs['X'],y=kwargs['y'], y_pred=kwargs['y_pred'])

        self.canvas.draw()

    def remove_surface(self):
        """Remove old decision surface."""
        if len(self.contours) > 0:
            for contour in self.contours:
                print("remove_surface" + str(type(contour)))
                if isinstance(contour, ContourSet):
                    print("remove_surface2" + str(type(contour)))
                    for lineset in contour.collections:
                        lineset.remove()
                else:
                    contour.remove()
            self.contours = []

    def remove_tube(self):
        if len(self.tube) > 0:
            for t in self.tube:
                print(type(t))
                if isinstance(t, matplotlib.collections.PathCollection):
                    t.remove()
                elif isinstance(t, list):
                    for l in t:
                        print(type(l))
                        l.remove()

            # TODO: Check memory leaks
            self.tube = []

    def plot_support_vectors(self, support_vectors):
        """Plot the support vectors by placing circles over the
        corresponding data points and adds the circle collection
        to the contours list."""
        cs = self.ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                             s=80, edgecolors="k", facecolors="none")
        self.contours.append(cs)

    def plot_support_vectors_svr(self, support_vectors_idx, X, y):
        if len(support_vectors_idx) > 0:
            cs = self.ax.scatter(X[support_vectors_idx], y[support_vectors_idx], s=75, c='r', edgecolors='r',
                   facecolors='none', linewidths=2)
            self.tube.append(cs)

    def plot_decision_surface(self, surface, type):
        X1, X2, Z = surface
        if type == 0:
            levels = [-1.0, 0.0, 1.0]
            linestyles = ['dashed', 'solid', 'dashed']
            colors = 'k'
            self.contours.append(self.ax.contour(X1, X2, Z, levels,
                                                 colors=colors,
                                                 linestyles=linestyles))
        elif type == 1:
            self.contours.append(self.ax.contourf(X1, X2, Z, 10,
                                                  cmap=matplotlib.cm.bone,
                                                  origin='lower', alpha=0.85))
            self.contours.append(self.ax.contour(X1, X2, Z, [0.0], colors='k',
                                                 linestyles=['solid']))
        else:
            raise ValueError("surface type unknown")

    def plot_epsilon_tube(self, epsilon, X, y, y_pred):

        index = np.argsort(X,axis=0)
        X = np.squeeze(np.array(X)[index])
        y_pred = np.squeeze(np.array(y_pred)[index])

        self.tube.append(self.ax.plot(X, y_pred, c='g', label='SVR model'))
        self.tube.append(self.ax.fill(np.concatenate([X, X[::-1]]),
                     np.concatenate([y_pred - epsilon,
                                     (y_pred + epsilon)[::-1]]),
                     alpha=.2, fc='b', ec='None', label='epsilon tube'))

class MenuBar(object):
    def __init__(self, root, controller):
        self.controller = controller
        menubar = Tk.Menu(root)

        # create a pulldown menu, and add it to the menu bar
        filemenu = Tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open data set", command=self.open_file)
        filemenu.add_command(label="Save data set", command=self.save_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = Tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=hello)
        menubar.add_cascade(label="Help", menu=helpmenu)

        # define options for opening or saving a file
        self.file_opt = options = {}
        options['defaultextension'] = '.dat'
        options['filetypes'] = [('all files', '.*'), ('SVMlight  files', '.dat'), ('comma separated values', '.csv'), ('Weka files', '.arff')]
        options['initialdir'] = './'
        options['initialfile'] = 'mydataset.dat'
        options['parent'] = root
        options['title'] = 'Choose file name and format'

        # display the menu
        root.config(menu=menubar)

    def open_file(self):
        file_name = tkFileDialog.askopenfilename(**self.file_opt)
        self.controller.open_file(file_name)


    def save_file(self):
        file_name = tkFileDialog.asksaveasfilename(**self.file_opt)
        self.controller.save_file(file_name)


class ControllBar(object):
    def __init__(self, root, controller):

        # Hyper-parameters form
        fm = Tk.Frame(root)
        kernel_group = Tk.Frame(fm)
        Tk.Radiobutton(kernel_group, text="Linear", variable=controller.kernel,
                       value=0, command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(kernel_group, text="RBF", variable=controller.kernel,
                       value=1, command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(kernel_group, text="Poly", variable=controller.kernel,
                       value=2, command=controller.refit).pack(anchor=Tk.W)
        kernel_group.pack(side=Tk.LEFT)

        valbox = Tk.Frame(fm)
        controller.complexity = Tk.StringVar()
        controller.complexity.set("1.0")
        c = Tk.Frame(valbox)
        Tk.Label(c, text="C:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(c, width=6, textvariable=controller.complexity).pack(
            side=Tk.LEFT)
        c.pack()

        controller.gamma = Tk.StringVar()
        controller.gamma.set("0.01")
        g = Tk.Frame(valbox)
        Tk.Label(g, text="gamma:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(g, width=6, textvariable=controller.gamma).pack(side=Tk.LEFT)
        g.pack()

        controller.epsilon = Tk.StringVar()
        controller.epsilon.set("5")
        g = Tk.Frame(valbox)
        Tk.Label(g, text="epsilon:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(g, width=6, textvariable=controller.epsilon).pack(side=Tk.LEFT)
        g.pack()
        valbox.pack(side=Tk.LEFT)

        valboxpol = Tk.Frame(fm)
        controller.degree = Tk.StringVar()
        controller.degree.set("3")
        d = Tk.Frame(valboxpol)
        Tk.Label(d, text="degree:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(d, width=6, textvariable=controller.degree).pack(side=Tk.LEFT)
        d.pack()

        controller.coef0 = Tk.StringVar()
        controller.coef0.set("0")
        r = Tk.Frame(valboxpol)
        Tk.Label(r, text="coef0:", anchor="e", width=7).pack(side=Tk.LEFT)
        Tk.Entry(r, width=6, textvariable=controller.coef0).pack(side=Tk.LEFT)
        r.pack()
        valboxpol.pack(side=Tk.LEFT)

        cmap_group = Tk.Frame(fm)
        Tk.Radiobutton(cmap_group, text="Hyperplanes",
                       variable=controller.surface_type, value=0,
                       command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(cmap_group, text="Surface",
                       variable=controller.surface_type, value=1,
                       command=controller.refit).pack(anchor=Tk.W)

        cmap_group.pack(side=Tk.LEFT)

        reg_group = Tk.Frame(fm)
        Tk.Radiobutton(reg_group, text="Classification",
                       variable=controller.classification, value=1,
                       command=controller.refit).pack(anchor=Tk.W)
        Tk.Radiobutton(reg_group, text="Regression",
                       variable=controller.classification, value=0,
                       command=controller.refit).pack(anchor=Tk.W)

        reg_group.pack(side=Tk.LEFT)

        train_button = Tk.Button(fm, text='Fit', width=5,
                                 command=controller.fit)
        train_button.pack()
        fm.pack(side=Tk.LEFT)
        Tk.Button(fm, text='Clear', width=5,
                  command=controller.clear_data).pack(side=Tk.LEFT)


def hello():
    print("hello!")


def get_parser():
    from optparse import OptionParser

    op = OptionParser()
    op.add_option("--output",
                  action="store", type="str", dest="output",
                  help="Path where to dump data.")
    return op


def main(argv):
    op = get_parser()
    opts, args = op.parse_args(argv[1:])
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("Scikit-learn Libsvm GUI")
    view = View(root, controller)
    model.add_observer(view)
    Tk.mainloop()

    if opts.output:
        model.dump_svmlight_file(opts.output)


if __name__ == "__main__":
    main(sys.argv)
