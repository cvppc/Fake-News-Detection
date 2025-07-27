Fake News Detection System which uses RoBERTa for classification and facilitated explainability with LIME and SHAP.

Contains static, template for front end and app.py too.

Conference paper, PowerPoint presentation and documentation have also been uploaded as a record for future reference.

Open the ipynb file, download it, upload the dataset to a Colab notebook, and run each cell, everything is good to go. But if you want to see the output in the UI, just download the app.py file which contains the Flask code, along with static and templates folders. Change the path inside the code if needed. Then, double-click the app.py file, a local server link will appear. You have to click on that link and it will take you to the UI. There, you can test some texts and see the predictions along with LIME and SHAP explanations.

You can save the trained model and also use it to predict texts, but the model is still not performing well. So we have added a pretrained model named cyberlord (etc. etc.), and the project uses it for prediction.

It’s just a single line, even in Colab, that is used for prediction and the same applies to the UI. We didn’t use our model for predictions, we just trained it with our dataset. But all predictions are made using the pretrained model "cyberlord" (you can find the proper name inside the code).
