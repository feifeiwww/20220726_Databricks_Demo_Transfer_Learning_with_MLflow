# Databricks Demo: Transfer Learning with MLflow 
## with FourthBrain

We will go hands-on with an image classification of Cats vs Dogs demo using **transfer learning**, while leveraging **MLflow** to track our model experiments on **Databricks**.

You may find more information and a recording about this demo on [FourthBrain website](https://discover.fourthbrain.ai/live-session/databricks?utm_campaign=Databricks%20Event&utm_medium=email&_hsmi=2&_hsenc=p2ANqtz--F8VKHIPHNwly3IwFlBZT7uYi4Jn3-fqVCD3M9GJl2h8qjWSSemEn5fAiN0DF7uY7krt5DdxtgPo6hf6YqQX19orXAIw&utm_content=2&utm_source=hs_email).

## If you would like run this demo, please download the 2 notebook files before running them in your workspace:

**1. You may directly view the code from the notebooks inside of the `databricks_version` folder on Github;** 


**2. Or run them on Databricks Community Edition (it is free. note: it is different from the Databricks Free Trial version):**

  (a) First sign up for a free Community Edition account, please refer to this [doc](https://docs.databricks.com/getting-started/community-edition.html) if you have questions. You can use [this link](https://community.cloud.databricks.com/login.html) to sign in if you have an existing account.
![image](https://user-images.githubusercontent.com/109642474/180575265-ecbf6401-bf87-4fa3-b769-965318ff1790.png)


  (b) Download [this file](https://github.com/feifeiwww/20220726_Databricks_Demo_Transfer_Learning_with_MLflow/blob/main/databricks_version/Transfer%20Learning%20Demo%20Part%201.ipynb) "Transfer Learning Demo Part 1.ipynb" insde of `databricks_version` folder on this repo, and import this .ipynb file into your user account using Databrick Community Edition. Refer to this [doc](https://docs.databricks.com/notebooks/notebooks-manage.html#import-a-notebook) under *"import a notebook"* section.
  
  ![image](https://user-images.githubusercontent.com/109642474/180575795-0e705ec3-4281-49b3-973e-630606c6adee.png)


  (c) Create a cluster under the "compute" tab on the left, select Databricks runtime version 10.4ML or up, let's name it `test10.4ML`. (note: use 10.4ML, not 10.4). Refer to this [doc](https://docs.databricks.com/clusters/create.html) about how to create a cluster. It may take a few minutes to start a cluster. 
  
  ![image](https://user-images.githubusercontent.com/109642474/180576008-c55d3162-a5df-414a-839c-7048c9af40b5.png)


  (d) Open the imported notebooks on Databricks from the workspace tab, open "transfer learning demo part1" notebook, attach the `test10.4ML` cluster to your notebook, then click "Run All" on the top to run the notebook. Refer to this [doc](https://docs.databricks.com/notebooks/notebooks-manage.html#attach-a-notebook-to-a-cluster) under the section *"Attach a notebook to a cluster"*. 
  
  ![image](https://user-images.githubusercontent.com/109642474/180576291-1bdcd11a-c400-4152-afe1-c92a8fc577c2.png)
  
  ![image](https://user-images.githubusercontent.com/109642474/180576518-f4f71fda-05e0-48b5-8a6a-d32048981d11.png)



  (e) You can also try to import and open "transfer learning demo part2" notebook similarly, attach it to the cluster `test10.4ML`, and click "Run All". It may take several minutes to run this notebook.

  (f) Important note: you must attach your notebook to a cluster before you can run it. If your cluster is terminated, you need to restart it or re-create a new cluster. 

  (g) You may check your MLflow experiments by clicking "Experiments" on the top. 

**3. Or on Databricks enterprise edition:**

* Download the .ipynb notebooks insde of `databricks_version` folder on this repo, and import them into your Databricks enterprise user account.

* Run the notebooks on a cluster with Databricks runtime version 10.4ML or up. (note: use 10.4ML, not 10.4)

* You may check your MLflow experiments results under the "Experiments" tab on the left, or inside of your notebook on the top under "Experiments". 

* In addition, you may uncomment the model registry optional code at the end of "transfer learning demo part2.ipynb", and run them. Check model registry results under "Models" tab.


**4. Or locally (for example, jupyter notebook):**

* Please follow [this](https://www.tensorflow.org/tutorials/images/transfer_learning) link for tensorflow transfer learning tutorial.
* To use MLflow locally, you may need to add additional steps for configuration and code. These steps may include but not limited to ```pip install mlflow``` for installing MLflow locally,  ```!mlflow ui``` for viewing the MLflow UI, and ```!pkill -f gunicorn``` for stopping the UI. Note: it is an excersie for the users, so the full code is not provided here. 
