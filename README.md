
# Beyond the Buzz

A brief description of what this project does and who it's for


## Contributions

**Naman Gupta(220686)**

## Data Preprocessing

Given the file 'train.csv', we plotted the data separately for each of the 9 parameters and found outliers that may reduce the accuracy of our model.


Raw Data | Processed Data
--- | ---
![Alt text](https://i.ibb.co/St5fH13/newplot.jpg) | ![Alt text](https://i.ibb.co/gyXwzCy/newplot-3.jpg)
![Alt text](https://i.ibb.co/fvqWqQ1/newplot-1.jpg) | ![Alt text](https://i.ibb.co/Rv20CkT/newplot-4.jpg)

To filter the data we used z-score as a parameter.
Whenever the z-score of a datapoint was either greater than 3 or less than -3, we scraped the data. Here, train is pandas dataframe and stats.zscore is imported from scipy library.

``` python
for i in params:
  train = train[abs(stats.zscore(train[i]))<=3]
```

Before normalization, the train dataframe (top 5 rows) looked like - 

|index|VERDICT|PARAMETER\_1|PARAMETER\_2|PARAMETER\_3|PARAMETER\_4|PARAMETER\_5|PARAMETER\_6|PARAMETER\_7|PARAMETER\_8|PARAMETER\_9|
|---|---|---|---|---|---|---|---|---|---|---|
|0|1|39353|85475|117961|118300|123472|117905|117906|290919|117908|
|1|1|17183|1540|117961|118343|123125|118536|118536|308574|118539|
|2|1|36724|14457|118219|118220|117884|117879|267952|19721|117880|
|3|1|36135|5396|117961|118343|119993|118321|240983|290919|118322|
|4|1|42680|5905|117929|117930|119569|119323|123932|19793|119325|


Finally, we normalisized the data so that it has a mean of 0 and a standard deviation of 1. Here, trainprocessed is the filtered dataset.

``` python
scaler = StandardScaler()
train_processed[params] = scaler.fit_transform(train_processed[params])
```

 We got - 

|index|VERDICT|PARAMETER\_1|PARAMETER\_2|PARAMETER\_3|PARAMETER\_4|PARAMETER\_5|PARAMETER\_6|PARAMETER\_7|PARAMETER\_8|PARAMETER\_9|
|---|---|---|---|---|---|---|---|---|---|---|
|0|1|-0\.06970642898763442|2\.2884185393973593|0\.09751489082811729|-0\.11849269255364434|0\.8139514260170707|-0\.3081528224112156|-0\.7225140110922403|1\.0456337456753682|-0\.6103052796765416|
|1|1|-0\.8849008589800971|-0\.799671007066648|0\.09751489082811729|-0\.0706434921686976|0\.7336061551539824|-0\.2159706704018587|-0\.7132748606465639|1\.2206836867401167|-0\.3318021068751961|
|2|1|-0\.16637517758710146|-0\.3244359719600109|0\.1605620090515678|-0\.20751446071168478|-0\.4799084402161773|-0\.3119511361231701|1\.4779583187045469|-1\.6433038876793558|-0\.6226635821146361|
|3|1|-0\.18803280136994135|-0\.6578032130454982|0\.09751489082811729|-0\.0706434921686976|0\.008414834280229261|-0\.24737980301994386|1\.082449353038754|1\.0456337456753682|-0\.42757895077042907|
|4|1|0\.052627890750070724|-0\.6390763699031835|0\.08969509321900714|-0\.5302183702845814|-0\.08975921426717537|-0\.10099863612077489|-0\.6341408038134363|-1\.6425900051669422|0\.01511309727988711|

We then check the correlation between the parameters, as it is clear from the obtained matrix that there is no correlation between any of the variables.

|index|VERDICT|PARAMETER\_1|PARAMETER\_2|PARAMETER\_3|PARAMETER\_4|PARAMETER\_5|PARAMETER\_6|PARAMETER\_7|PARAMETER\_8|PARAMETER\_9|
|---|---|---|---|---|---|---|---|---|---|---|
|VERDICT|1\.0|0\.007374874140603658|-0\.019030255915717997|-0\.014258449027845751|-0\.020536733558475156|-0\.009925964011483827|0\.006333060608162153|0\.009890192410095809|0\.003442055686773596|0\.010322844346048021|
|PARAMETER\_1|0\.007374874140603658|1\.0|0\.010860591634781303|-0\.057794658984812536|-0\.007577121962256353|0\.0009147055282313084|-0\.02347419475193103|0\.009076361819325739|0\.05645086639110787|0\.004495343750986927|
|PARAMETER\_2|-0\.019030255915717997|0\.010860591634781303|1\.0|-0\.07671399120182355|0\.031834285925097724|-0\.06672360673094148|0\.040485155310006404|-0\.045797874184217886|-0\.1550985266629161|-0\.025561490682054946|
|PARAMETER\_3|-0\.014258449027845751|-0\.057794658984812536|-0\.07671399120182355|1\.0|0\.23690188383288915|0\.03216095051930841|0\.01524605135333103|0\.07514775886987143|-0\.04533840533278488|0\.0017472577315428624|
|PARAMETER\_4|-0\.020536733558475156|-0\.007577121962256353|0\.031834285925097724|0\.23690188383288915|1\.0|0\.0755539000007956|-0\.011615159191281978|0\.03935912107958597|0\.06898399553363642|0\.026635553646687377|
|PARAMETER\_5|-0\.009925964011483827|0\.0009147055282313084|-0\.06672360673094148|0\.03216095051930841|0\.0755539000007956|1\.0|-0\.013527466272678269|0\.06838376113331358|0\.0842034527574188|0\.06667492174678759|
|PARAMETER\_6|0\.006333060608162153|-0\.02347419475193103|0\.040485155310006404|0\.01524605135333103|-0\.011615159191281978|-0\.013527466272678269|1\.0|0\.039345835316920576|-0\.14520089333558753|0\.2700329859552825|
|PARAMETER\_7|0\.009890192410095809|0\.009076361819325739|-0\.045797874184217886|0\.07514775886987143|0\.03935912107958597|0\.06838376113331358|0\.039345835316920576|1\.0|-0\.18292874508943438|0\.18232907088495806|
|PARAMETER\_8|0\.003442055686773596|0\.05645086639110787|-0\.1550985266629161|-0\.04533840533278488|0\.06898399553363642|0\.0842034527574188|-0\.14520089333558753|-0\.18292874508943438|1\.0|-0\.22415365418284647|
|PARAMETER\_9|0\.010322844346048021|0\.004495343750986927|-0\.025561490682054946|0\.0017472577315428624|0\.026635553646687377|0\.06667492174678759|0\.2700329859552825|0\.18232907088495806|-0\.22415365418284647|1\.0

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
    matrics = ['accuracy']
)

model.fit(
    data,y_array,
    batch_size=200,
    epochs=140
)
```

NOTE: We tried using the traditional 80-20 split method for model traing but it kept failing due to the highly unbalanced datapoints grouped by verdict ie. legit and fraud transactions. So we dropped that thought.



## Results

The result while training the model are - 

Accuracy: 
Loss: 

And on investigating in detail we found out - 

- It correctly identified fraud transactions ____ out of _____ times.
- It correctly identified legit transactions ____ out of _____ times.

And the results on test.csv are posted in the repo
## Thanks:)
