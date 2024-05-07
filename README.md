This project uses federated learning to train a model to recognise hindi alphabets and number.
You can use server code as it is. And Strategy class created because I wanted print and see how aggregrations works.
Pickle is used to write final weights to a file which can be used for testing.
In client side you have to update  trainPath , testPath and Server's IP address.
And test file reads weights from aggregated_weights then runs the model for testing.
