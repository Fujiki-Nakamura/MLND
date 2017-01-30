<h1>Capstone Project</h1>
<h2>Machine Learning Engineer Nanodegree</h2>
<p align="right">Fujiki Nakamura</p>


<h2 align="center">Definition</h2>

<h3>Project Overview</h3>
<p>&nbsp;&nbsp;
In this project, I solve a problem provided by Kaggle.Kaggle is a Data Science competition platform and provide their competitors with a lot of data in real world. Many companies are working with Kaggle and sharing their data on the platform of Kaggle to discover the method that produce the best result and the talented competitors. Competitors at Kaggle make the most of their Data Analysis and Machine Learning skills and aim to win their competitions. Kaggle is also platform of learning practical skills of Data Science and helps many junior Data Scientists to advance their skills providing discussions on the forum and open scripts.
</p>
<p>&nbsp;&nbsp;
The problem I solve in this project is provided by AllState corporation, which is [the second largest personal insurer in the United States](https://en.wikipedia.org/wiki/Allstate). They provide their data including the information of how severe an insurance claim is and other information that can be useful to predict the severity. According to the Kaggle home page of this competition, they are [currently developing automated methods of predicting the cost, and hence severity, of claims](https://www.kaggle.com/c/allstate-claims-severity)  and looking for the talented competitors to recruit.
</p>
<p>&nbsp;&nbsp;
The objective for this project is not recruitment, but through this problem I provide a case study of developing automated methods to produce good predictions.
</p>

<h3>Problem Statement</h3>
<p>&nbsp;&nbsp;
This is a problem of regression because the target value (, which is the severity here) is numerical and a problem of supervised learning because the target value is explicitly provided in the training dataset and we have to predict the scores for the test dataset. In such a kind of problem, I need to build the model which predicts the cost of claims that Allstate's customers have to pay as correctly as possible. In order to achive that goal, I follow the strategy outlined below:

<ol type="1">
  <li>
    Data Exploration:
    </br>explore the data and grasp how it looks like. In this process, I examine the meanings and distibutions of the data.
  </li>
  <li>
    Data Preprocessing:
    </br>preprocess the data in a way the each model built later can handle them appropriately. Convert some type of values into another type of values to feed them into the models appropriately, transform some values to correct their skewness in distributions and remove some unnecessary values.
  </li>
  <li>
    Building Machine Learning models:
    </br>Develop several Machine Learning models to approach the solution. Build the baseline models and define their performance as the criteria of improvement later.
  </li>
  <li>
    Tuning the models:
    </br>enhance the performence of the models built in step 3. Tune the hyperparameter of the models, make the better architecture of the models and so on.
  </li>
  <li>
    Ensembling the models:
    </br>To achive the better performance than the single models above, ensumble them introduce the method of stacking.
  </li>
</ol>

</p>

<h3>Metrics</h3>
<p>&nbsp;&nbsp;
The metric to measure the performences is [Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error). In addition to the fact that the competition requires MAE as the metric, MAE is appropriate because it measures the difference between predicted values and ground truth values directly. Of course, other metrics like MSE or RMSE seem appropriate but they have a "side effect" which we don't want here. When they square the differences, they make bigger errors bigger and smaller errors smaller (More precisely, when squared, errors more than 1 have bigger impact on the total error and errors less than 1 have smaller). MAE is one of the appropriate metrics to measure how much the difference is between the monetary values and to handle all errors equally, and interpreting the difference is intuitively easy (e.g How much differs from the true cost).
</p>

<h2 align="center">Analysis</h2>

<p>&nbsp;&nbsp;
</p>
