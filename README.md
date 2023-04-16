
# Beyond the Buzz

A brief description of what this project does and who it's for


## Contributions

**Trijal Srivastava, Naman Gupta**

Hi and welcome to our submission!
We will try our best to explain the logic, implementation and code. The code represents our final solution, but we will try to give you some insight into how we reached this.

## Data Preprocessing

Given the file 'train.csv', we plotted the data separately for each of the 9 parameters and found outliers that may reduce the accuracy of our model.


Raw Data | Processed Data
--- | ---
![Alt text](https://i.ibb.co/St5fH13/newplot.jpg) | ![Alt text](https://i.ibb.co/gyXwzCy/newplot-3.jpg)
![Alt text](https://i.ibb.co/fvqWqQ1/newplot-1.jpg) | ![Alt text](https://i.ibb.co/Rv20CkT/newplot-4.jpg)

To filter the data we used z-score as a parameter.
Whenever the z-score of a datapoint was either greater than 3 or less than -3, we scraped the data.

Finally, we normalisized the data so that it has a mean of 0 and a standard deviation of 1.

## Building a Model

Our task now is to build an effective neural network model that will be capable of telling fraud transactions with accuracy.

```python
model = Sequential([               
        tf.keras.Input(shape=(9,)),    
        Dense(64,activation ='relu'),
        Dense(256,activation ='relu'),
        Dense(64,activation ='relu'),
        Dense(1,activation ='sigmoid')
    ])
```
Now we have modeled the neural nets. They are powerful universal function approximators, and amongst all the models we used this one, gave the best results.

We found the best architecture for our Neural Network by trial-and-error (not shown here). We observed that having 3 layers improved both the training and validation loss as compared to 2 layers, and adding an extra 4th layer worsened the validation loss due to overfitting of data. The number of neurons in each layer was determined by trying several variations. On making 256 neurons in one of the layers, we were finally able to estimate satisfactorily.


We tried several learning rates and settled on 0.001 because decreasing it further was significantly increasing the training time without much improvement on the results.

``` python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    data,y_array,
    batch_size=200,
    epochs=120
)
```
