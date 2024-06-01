# Speed_Dating_DataAnalysis

Analyzing the data in Speed_date_data
Using data from kaggle.
https://www.kaggle.com/datasets/mexwell/speed-dating/data

Using cleaned_speed_data by performing a cleaning data process on the data in advance

Code that changes the parameters of the classification model and finds the best combination using various Scaling and Encoding methods.
We use 2 Encoding method(One-hot Encoding / Label Encoding), 2 Scaler (MinMax / Robust), 2 Parameter of classifier model(k = 3, 5)
so the number of total case is 8.


# Instructions
Files exist for cases 1 through 8.
When each file is executed, it outputs a data inspection process and proceeds with the preprocessing process.
At this time, each file has a different method of scaling and encoding.
In the Kn classifier process, the accuracy can be obtained by converting the k value.
+)
Knn allows us to predict about new data.
We also do the k-means cluster process
The evaluation for each model proceeds at the end.

# Data Information
gender(integer) : Gender of the person evaluated.  female is 0 / male is 1
age(integer): age of the person evaluated
income(float): income of the person evaluated
career(object): carrer of the person evaluated
dec(int): whether this individual was a match (rater perspective)
attr(float): attractiveness of the person evaluated by rater
sinc(float): sincerity of the person evaluated by rater
intel(float):  intelligence of the person evaluated by rater
fun(float): fun of the person evaluated by rater
amb(float): ambitiousness of the person evaluated by rater
shar(float): degree of shared interest of the person evaluated by rater
like(float): overall rating
prob(float): whether the rater believed that interest would be reciprocated
met(int):whether the two had met prior to the speed date
-> Because We cleaning data already,  most data types change from integers to floats.
