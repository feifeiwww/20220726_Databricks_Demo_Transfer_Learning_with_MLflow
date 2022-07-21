# 20220726_Databricks_Demo_Transfer_Learning_with_MLflow
We will go hands-on with an image classification of cats VS dogs demo using ***transfer learning***, while leveraging ***MLflow*** to track our model experiments on ***Databricks***.

You may find more information and a recording about this demo on [FourthBrain website](https://discover.fourthbrain.ai/live-session/databricks?utm_campaign=Databricks%20Event&utm_medium=email&_hsmi=2&_hsenc=p2ANqtz--F8VKHIPHNwly3IwFlBZT7uYi4Jn3-fqVCD3M9GJl2h8qjWSSemEn5fAiN0DF7uY7krt5DdxtgPo6hf6YqQX19orXAIw&utm_content=2&utm_source=hs_email).

If you would like Run this demo:

**1. Locally (for example, jupyter notebook):**

* Please follow [this](https://www.tensorflow.org/tutorials/images/transfer_learning) link for tensorflow transfer learning tutorial.

**2. Or on Databricks Community Edition (it is free. note: it is different from the Databricks Free Trial version):**

(a) First sign up for a free Community Edition account, please refer to this [doc](https://docs.databricks.com/getting-started/community-edition.html) if you have questions. You can use [this link](https://community.cloud.databricks.com/login.html) to sign in if you have an existing account.

(b) Then download [this file](https://github.com/feifeiwww/20220726_Databricks_Demo_Transfer_Learning_with_MLflow/blob/main/databricks_version/feifei_transfer_learning_with_MLflow_demo.dbc) "feifei_transfer_learning_with_MLflow_demo.dbc"  insde of `databricks_version` folder on this repo, and import this .dbc file into your user account using Databrick Community Edition. Refer to this [doc](https://docs.databricks.com/notebooks/notebooks-manage.html#import-a-notebook) under *"import a notebook"* section.

(c) Create a cluster under the "compute" tab on the left, select Databricks runtime version 10.4ML or up. (note: use 10.4ML, not 10.4). Refer to this [doc](https://docs.databricks.com/clusters/create.html) about how to create a cluster. It may take a few minutes to start a cluster. 

(d) Open the imported notebooks on Databricks from the workspace tab, open "transfer learning demo part1" notebook, attach the cluster to your notebook, then click "Run All" on the top to run the notebook. Refer to this [doc](https://docs.databricks.com/notebooks/notebooks-manage.html#attach-a-notebook-to-a-cluster) under the section *"Attach a notebook to a cluster"*. 

(e) You can now try to open "transfer learning demo part2" notebook, attach it to the 10.4ML cluster, and click "Run All". It may take several minutes to run this notebook.

(f) Important note: you must attach your notebook to a cluster before you can run it. If your cluster is terminated, you need to restart it or re-create a new cluster. 

(g) You may check your MLflow experiments by clicking "Experiments" on the top. 

**3. Or on Databricks enterprise edition:**

* Download [this file](https://github.com/feifeiwww/20220726_Databricks_Demo_Transfer_Learning_with_MLflow/blob/main/databricks_version/feifei_transfer_learning_with_MLflow_demo.dbc) "feifei_transfer_learning_with_MLflow_demo.dbc" insde of `databricks_version` folder on this repo, and import this .dbc file into your Databricks user account.

* Run the notebooks on a cluster with Databricks runtime version 10.4ML or up. (note: use 10.4ML, not 10.4)

* You may check your MLflow experiments results under the "Experiments" tab on the left, or inside of your notebook on the top under "Experiments". 

In addition, you could view the code directly from the `html_versin` folder. 
