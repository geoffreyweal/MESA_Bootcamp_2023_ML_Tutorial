from ipywidgets import interact, widgets
import numpy as np
import pylab as pl
from matplotlib.lines import Line2D

point_size = 35.0

def lighten_rgb_color(rgb_colour, amount):
    return (min(rgb_colour[0] +amount, 1.0), min(rgb_colour[1] +amount, 1.0), min(rgb_colour[2] +amount, 1.0))

def make_knn_plot(sepal_length, sepal_width, n_neighbors, prediction_type, X, y, Z, xx, yy, sepal_length_lims, sepal_width_lims, cmap_light, cmap_bold, knn):

    pl.figure()
    if prediction_type == 'definite':
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
        z_predict = knn.predict([(sepal_length, sepal_width)])[0]
    elif prediction_type == 'probabilistic':
        pl.pcolormesh(xx, yy, Z)
        #z_predict = knn.predict_proba([(sepal_length, sepal_width)])[0]

    # Plot also the training points
    pl.scatter(X[:, 0], X[:, 1], s=point_size, c=y, cmap=cmap_bold)
    pl.xlabel('sepal length (cm)')
    pl.ylabel('sepal width (cm)')
    pl.axis('tight')
    pl.xlim(sepal_length_lims)
    pl.ylim(sepal_width_lims)

    # feature vector for sepal length and width
    point_fv = np.array([sepal_length, sepal_width])

    # Determine the nearest neighbour based on the euclidean distance
    nearest_neighbours = []
    for jj in range(len(X)):
        diff = point_fv - X[jj][:2]
        distance = (sum(diff ** 2.0)) ** 0.5
        if len(nearest_neighbours) < n_neighbors:
            nearest_neighbours.append((distance,jj))
        else:
            for nni in range(len(nearest_neighbours)):
                nni_dist = nearest_neighbours[nni][0]
                if distance < nni_dist:
                    nearest_neighbours.insert(nni,(distance,jj))
                    del nearest_neighbours[-1]
                    break
        nearest_neighbours.sort()

    # Add lines to indicate the k-nearest neighbours from point index
    iris_counter = [0,0,0]
    plot_data = []
    for _, nni in nearest_neighbours:
        feature_vector_nni = X[nni]
        plot_data.append((point_fv[0], feature_vector_nni[0]))
        plot_data.append((point_fv[1], feature_vector_nni[1]))
        plot_data.append('k-')
        iris_counter[y[nni]] += 1

    # Plot the k-nearest neighbours prediction data
    pl.plot(*plot_data)
    if prediction_type == 'definite':
        pl.scatter((point_fv[0],), (point_fv[1],), s=point_size*0.5, marker='x', color=cmap_bold(z_predict), zorder=2001)
    pl.scatter((point_fv[0],), (point_fv[1],), s=point_size*2, marker='x', c='k', zorder=2000)

    if prediction_type == 'probabilistic':
        iris_counter = np.array(iris_counter)/sum(iris_counter)
        iris_counter = [str(round(value*100.0,1))+' %' for value in iris_counter]

    # Give the legend details
    legend_elements = [Line2D([0], [0], marker='o', color=cmap_bold(0), label='Setosa: '+str(iris_counter[0]),      linewidth=0, zorder=1000),
                       Line2D([0], [0], marker='o', color=cmap_bold(1), label='Versicolour: '+str(iris_counter[1]), linewidth=0, zorder=1000),
                       Line2D([0], [0], marker='o', color=cmap_bold(2), label='Virginica: '+str(iris_counter[2]),   linewidth=0, zorder=1000)]
    pl.legend(handles=legend_elements, frameon=True)
    pl.show()

def visualise_knn(sepal_length_lims, sepal_width_lims, n_neighbors, prediction_type, X, y, Z, xx, yy, cmap_light, cmap_bold, knn):
    # Create interactive figure, containing lines for the k neighest neighbours about point i
    def interactive_knn_figure(sepal_length=sepal_length_lims[0], sepal_width=sepal_width_lims[0]):
        make_knn_plot(sepal_length, sepal_width, n_neighbors, prediction_type, X, y, Z, xx, yy, sepal_length_lims, sepal_width_lims, cmap_light, cmap_bold, knn)
    if prediction_type not in ['definite', 'probabilistic']:
        raise Exception('Error: prediction_type must be either "definite" or "probabilistic". prediction_type = '+str(prediction_type))
    interact(interactive_knn_figure, sepal_length=widgets.FloatSlider(min=sepal_length_lims[0], max=sepal_length_lims[1], step=0.01, value=sepal_length_lims[0]), sepal_width=widgets.FloatSlider(min=sepal_width_lims[0], max=sepal_width_lims[1], step=0.01, value=sepal_width_lims[0]))


# =======================================================================================================================

from sklearn.neighbors import KNeighborsClassifier
def make_multi_nn_figure(prediction_type, sepal_length, sepal_width, X, y, cmap_light, cmap_bold):

    # Make figure
    fig, axes = pl.subplots(3, 3, figsize=(10, 10))

    # Create variables for making a meshgrid for showing decision areas by knn
    x_min, x_max = (4.0, 8.0) #X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = (2.0, 4.5) #X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Perform knn algorithm for several values of n_neighbors
    n_neighbors = 1
    for row in range(3):
        for col in range(3):

            # Select axis
            axis = axes[row][col]

            # Create the model
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

            # Fit the model
            knn.fit(X, y)

            # Predict all the iris results for a range of sepal widths and lengths
            if prediction_type == 'definite':
                Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            if prediction_type == 'probabilistic':
                Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            if prediction_type == 'definite':
                Z = Z.reshape(xx.shape)
                axis.pcolormesh(xx, yy, Z, cmap=cmap_light)
            if prediction_type == 'probabilistic':
                Z = Z.reshape(list(xx.shape)+[3])
                for i1 in range(len(Z)):
                    for i2 in range(len(Z[i1])):
                        Z[i1][i2] = lighten_rgb_color(Z[i1][i2], 0.65)
                axis.pcolormesh(xx, yy, Z)

            # Plot also the training points
            axis.scatter(X[:, 0], X[:, 1], s=point_size*0.6, c=y, cmap=cmap_bold)
            if (row == 2) and (col == 1):
                axis.set_xlabel('sepal length (cm)')
            if (row == 1) and (col == 0):
                axis.set_ylabel('sepal width (cm)')
            #axis.axis('tight')

            axis.legend(handles=[Line2D([0], [0], color='w', marker=',', markersize=0, lw=0, label='nn = '+str(n_neighbors))], frameon=True)

            n_neighbors += 1
    pl.show()

def visualise_multi_knn(sepal_length, sepal_width, X, y, cmap_light, cmap_bold):
    # Create interactive figure, containing lines for the k neighest neighbours about point i
    def interactive_mutli_nn_figure(prediction_type='definite'):
        make_multi_nn_figure(prediction_type, sepal_length, sepal_width, X, y, cmap_light, cmap_bold)
    interact(interactive_mutli_nn_figure, prediction_type=['definite', 'probabilistic'])









