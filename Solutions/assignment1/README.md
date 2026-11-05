# Notes for Assignment 1 and Libraries

## Assignment 1
- **K-Nearest Neighbors (k-NN)** with the loss function is the L2 distance, training by memorizing data into the model, testing by outputting the label has smallest L2 distance.
- **Cross Validation** to choose the best hyperparameter k in k-NN.
- **Softmax Loss Function:** Given $\mathbf{s} = f(x_i; W) \in \mathbb{R}^C$, we define
$$L(W) = \sum_{i = 1}^{n} -\log (\Pr[Y = y_i \, | \, X = x_i]) = \sum_{i = 1}^{n} - \log \left(\dfrac{e^{s_{y_i}}}{\sum_{j \in [C]} e^{s_j}} \right).$$
- **Multiclass SVM Loss Function:** Given $\mathbf{s} = f(x_i; W) \in \mathbb{R}^C$, we define
$$L(W) = \sum_{i=1}^{n} \sum_{j \neq y_i} \max\{0, s_j + \Delta - s_{y_i}\}.$$
- Define a class for our ML/DL model:
```python
class OurModel:
    def __init__(self, ...):
        pass
    
    def train(self, X, y, ...):
        pass

    def predict(self, X):
        pass
    
    def loss(self, X_batch, y_batch):
        pass
    
    def save(self, fname): # Save model params
        pass

    def load(self, fname): # Load model params
        pass
```
- **Multi-Layer DL:** Forward pass needs to store cache (intermediate values when computing forward) for the backward propagation.
- **Feature Extraction:** Image → Feature → Model → Output. DL model takes these features as inputs rather than raw images.
- **Optimizers:** SGD with Momentum, RMSProp, Adam.

## Numpy Functions
- ``np.flatnonzero()``: Return all indices of non-zero elements after flattening the input
- ``np.argsort()``: Return a 1D-array of indices that would sort an array
- ``np.bincount(a)``: Return a 1D-array of $\max_i a[i] + 1$ elements that counts number of occurences of each non-negative value in array
    - ``np.bincount(np.array([0, 1, 1, 3, 2, 1, 7])): return array([1, 3, 1, 1, 0, 0, 0, 1])``
    - 0 occurs zero time, 1 occurs three times, ...
- ``np.argmax()``: Return the index of the maximum value
- ``np.concatenate((), axis=)``: Return the concatenated matrix  defined in parentheses with the predifined axis.
- ``np.array_split(a, num, axis=)``: Return ``num`` sub-arrays of ``a``
- ``np.hstack((a, b))``: Return the column-wise concatenated array
- ``np.outer(a, b)``: Return the outer product of two input vectors