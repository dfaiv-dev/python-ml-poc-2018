This code is proof of concept work I did for feasibility of using Machine Learning in Precision Agriculture. I worked on this project from 2017-2018 on my own time while also meeting my full time full stack software engineering job demands. I had no experience with Python, machine learning, Docker or Bash scripting when I started.

#### Summary
* SciKit Learn, XGBoost, [H2O.ai](https://www.h2o.ai/products/h2o-automl/) based models (see `/src/modeling/scripts`)
* Data export, cleaning and feature engineering (see `/src/data`)
* Automated infrastructure and pipeline management (see `/infrastructure`)
    * provision and setup VMs
    * build and deploy experiments as Docker images
    * auto shutdown and deallocate VMs when experiments were done
    * results stored in cloud blob storage 

#### Code Layout
* `/src` - main code: modeling, feature engineering, utilities
    * `/modeling/scripts` - experiment scripts using different techniques and libraries
    * `/data` - data import, cleaning, feature engineering
* `/infrastructure` - scripts and Docker files to setup VMs, build experiment containers, etc
* `/_archived_versions` - invalidated (false data assumptions) and other abandoned experiments

#### Where I Would Invest Time and Resources in a V2
* A stronger background in stats would be very useful, and something I would invest in if this was my main professional focus.
* Run larger datasets, either on larger VMs or using a distributed clustered framework (TensorFlow, Data Bricks).
* Focus on feature engineering - the data was pretty raw, even with the cleaning, bucketing, etc.

#### Resources I used for Machine Learning (that I remember):
* [Andrew Ng on Coursera](https://www.coursera.org/learn/machine-learning)
* https://machinelearningmastery.com/  