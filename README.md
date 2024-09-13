# California Housing Price Prediction

### Table of Contents:

1. Project Overview
2. Key Features
3. Project Workflow
4. Skills demonstrated
5. Setup and Installation
6. Model Training and Results
7. Running the Web Application on LocalHost
8. Deployment on AWS and Azure

## Project Overview

This project demonstrates a complete machine learning pipeline, from data ingestion to model deployment. The goal is to predict the median housing value based on various features like housing median age, total rooms, population, etc., using a Flask web application for user interaction.

## Key Features

- **Automated Data Ingestion:** Automatically downloads a dataset from **_Kaggle_** using API calls and stores it as a CSV file locally. The data is also stored in **_MongoDB Atlas (NoSQL)_** for storage in cloud databases.
- **Data Preprocessing Pipeline:** The data preprocessing pipeline includes handling missing values, encoding categorical variables, scaling numerical features, and splitting the dataset into training and testing sets.
- **Model Selection using GridSearchCV:** Multiple machine learning models are trained and optimized using `GridSearchCV`. The best model is saved in a pickle file for future use.
- **Flask Web Application:** A user-friendly web interface built with **_HTML_** and **_CSS_** allows users to input features and receive predictions. The interface is designed to be intuitive and responsive, enhancing the user experience.
- **Cloud Deployment:** The project is designed for deployment on both **_Azure_** and **_AWS_** platforms, enabling scalable machine learning solutions.'

## Project Workflow

### 1. Data Ingestion
- **Kaggle API Integration:** The project automatically downloads the dataset from Kaggle using an API key.
- **Local and Cloud Storage:** After downloading, the data is saved locally as a CSV file and uploaded to MongoDB Atlas if it's not already stored there.

### 2. Data Preprocessing
The data preprocessing pipeline includes:
- Imputation
- Encoding categorical features (like `ocean_proximity`)
- Feature scaling
- Splitting data into training and testing sets

### 3. Model Training and Selection
- Multiple models are trained using `GridSearchCV` to find the best model for predicting the median house value in the districts of California.
- The selected model is then serialized and stored as a `.pkl` file, along with preprocessing, imputing, feature engineering and data transformation objects.

### 4. Web Application - HTML/CSS with Flask Framework
The Flask web application provides an easy interface for users to input features and receive predictions.

* Users enter the housing data through a web form.
* After submitting the form, the model predicts the potential median housing value.
* A results page displays the prediction along with a button to submit a another query.

### 5. Deployment on Azure and AWS

The application is containerized and deployed on both Azure and AWS for scalable, cloud-based machine learning services.

## Skills Demonstrated
* **Machine Learning:** Hands-on experience with various models, tuning hyperparameters using GridSearchCV, and model evaluation.
* **Data Science:** Automated data ingestion and preprocessing pipeline, handling large datasets, and integration with MongoDB Atlas.
* **Web Development:** Front-end development with HTML/CSS and back-end development using Flask.
* **Cloud Computing:** Cloud storage with MongoDB Atlas and deployment on Azure and AWS.
* **Automation:** End-to-end automation for data ingestion, preprocessing, model training, and deployment.

## Setup and Installation
### Prerequisites
* Python 3.11
* Anaconda or Miniconda Installed
* **Microsoft Azure** or **AWS** creditentials
* .env file with required secret keys

### Installation

#### Fork this repository:

* Navigate to the GitHub repository.
* Click on the "Fork" button in the top-right corner.
* Clone the forked repository to your local machine:
 ```
git clone https://github.com/sohbatSandhu/california-housing-price-prediction.git
```

#### Create Conda or Python Environment

```
conda create -p venv python=3.11
```

#### Install the required packages and dependencies:

```
pip install -r requirements.txt
```
Set up the Kaggle API and MongoDB Atlas credentials.

## Model Training and Results

Run the following command in the root of the current workplace
```
python src/components/data_ingestion.py
```

### Model Metrics

Below are the best models and their corresponding training scores:

Model | Training Score
------|-----------------
Linear Regression | 0.6726
Ridge Regression | 0.6727
Decision Tree Regressor | 0.7078
Random Forest Regressor | 0.8066
Bagging Regressor | 0.8194
Gradient Boosting Regressor | 0.8335
AdaBoost Regressor | 0.6345
XGBoost Regressor | 0.8418

#### **Best model: XGBoost Regressor**

Training Score: _0.8418_

Test Score: _0.8158_

## Running the Web Application on LocalHost

### Start the Flask application:
```
python application.py
```

Access the web application at http://127.0.0.1:5000/california-housing/predict.

### Example Run

![alt text](screenshots/housing-form.png)

![alt text](screenshots/prediction.png)

## Deployment on AWS and Azure

### 1.Fork the repository (see installation for instructions)

### 2. Use Dry-Run Options
* AWS and Azure offer ways to simulate or validate the deployment without actually provisioning resources:
    * **For AWS**, you can use `aws cloudformation validate-template` or similar commands in your workflow to validate the CloudFormation template or infrastructure setup.
    * **For Azure**, you can use `az deployment validate` to check ARM templates or Bicep configurations for correctness without deploying them.

### 3. Set Up workflows using AWS or Azure Credentials
Depending on your target cloud platform (AWS or Azure), follow these steps to set up secrets in your GitHub repository.

#### For AWS Deployment:

* Navigate to your GitHub repository.

* Click on `Settings > Secrets and variables > Actions > New repository secret`.

* Add the following secrets:
    * AWS_ACCESS_KEY_ID: Your AWS Access Key ID.
    * AWS_SECRET_ACCESS_KEY: Your AWS Secret Access Key.
* Set the region for AWS in the yml file:

```
aws-region: us-west-2  # Replace with your region
```

**For Azure Deployment:**

* Set up a service principal in Azure:
    * Run the following command in the Azure CLI to create credentials outputing a JSON object for copying:
```
az ad sp create-for-rbac --name "github-actions" --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} \
  --sdk-auth
```

Replace the JSON output in the `.github/workflows/azure-deployment.yml`

* In your GitHub repository, navigate to `Settings > Secrets > Actions > New repository secret`.
* Add a new secret:
    * Name: AZURE_CREDENTIALS
    * Value: The JSON object output from the Azure CLI.

### 4. (Optional) Using existing workflows files for Deployment

To deploy this configuration on their own machine using their own credentials, make the following changes:

**For AWS Deployment:**

1. **Set Up AWS CLI**

Install the AWS CLI on your machine and configure it with your AWS credentials.
```
aws configure
```
This will set up your **AWS Access Key, Secret Access Key,** and **Region.**

2. **AWS Access Keys as GitHub Secrets**

* Navigate to your GitHub repository.
* Go to `Settings > Secrets and Variables > Actions` and add the following secrets:
    * `AWS_ACCESS_KEY_ID`
    * `AWS_SECRET_ACCESS_KEY`
* These values will be used in the GitHub Actions pipeline to authenticate and deploy to AWS.

3. **Elastic Beanstalk Environment**

* Ensure you have an existing Elastic Beanstalk (EB) environment or let the script create one automatically.
* Adjust the `eb init` command if you have specific configurations for your EB environment:

```
eb init -p python-3.7 my-app --region your-region
```
* `my-app` is the name of your application, and `your-region` should be the AWS region where your application is hosted.

4. **EB Deploy Command**

If your app is already deployed to an environment, adjust the `eb create` command to deploy to an existing EB environment by using:
```
eb deploy your-environment-name
```
5. **Adjust Region and Application Name**

Change the `aws-region` and the Elastic Beanstalk application name (`my-app`) to match your actual settings in AWS.

**For Azure Deployment:**

1. **Azure Web App Creation**

First, create an Azure Web App. This can be done via the Azure portal, Azure CLI, or other methods. This will give them:
* App Name
* Publish Profile (credentials for deployment)

2. **Modifying the YAML File**
*  **App Name**
    * Change the app-name under the deploy section to their own Azure Web App name.
```
with:
  app-name: 'your-web-app-name'
  slot-name: 'Production'
```
* **Publish Profile**

    * Go to the Azure Portal.
    * Navigate to the newly created Web App.
    * In the Deployment Center, download the Publish Profile.
    * Add this publish profile to your GitHub repository's secrets.
* In the GitHub repository, go to `Settings > Secrets and variables > Actions > New repository secret`.
* Name the secret, `AZUREAPPSERVICE_PUBLISHPROFILE`.
* Paste the contents of the publish profile XML file into the `value` section and save it.

In the YAML file, modify the following section to reference the correct secret:
```
publish-profile: ${{ secrets.AZUREAPPSERVICE_PUBLISHPROFILE }}
```

### 5. Simulate or Prepared Deployment

Without incurring costs or triggering a live deployment, you can simulate or prepare the deployment without actually deploying the application. You can use any of the following ways:

**For AWS deploment:**

**1. Comment Out the `eb deploy` Command**

You can comment out the final deployment step in the deploy job like this:

This will stop the process from actually deploying but will still perform the build steps and create a deployment-ready zip file.

**2. Remove AWS Credentials Step**

If you don't need to authenticate for a simulated deployment, you can comment out or remove the AWS credentials configuration step.

By commenting this out, you avoid attempting to connect to AWS at all.

**3. Run Locally Using `workflow_dispatch`**

You can make sure that your workflow only runs when you manually trigger it, by using workflow_dispatch:

```
on:
  workflow_dispatch:
```
This will allow you to run the workflow only when you choose to, instead of on every push to the repository.

**For Azure deployment:**

**1. Comment Out or Remove the Deployment Step**

To comment out the deployment step, use # in front of each line of the `deploy` job. You can otherwise remove it completely.

This will skip the deployment but allow you to:

* Run the build steps.
* Test the setup locally.
* Ensure the virtual environment is created, dependencies are installed, and build artifacts are generated.
