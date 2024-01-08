# AWS-Number-Guesser

Webapp to "guess" the user's drawn input number through use of a created pytorch neural network hosted as an aws Sagemaker endpoint. 
Webapp is hosted on aws Amplify with backend contained entirely in aws services.

## Demo video
https://github.com/PiyushBud/AWS-Number-Guesser/assets/93603829/c30fd6de-e532-434d-923c-0812a600a1dd


## Flow
User draws a number and hits the 'guess' button. Webapp scales down the pixel grid to 28x28 for proper input into neural network model.
Pixel data is then packaged into a JSON and sent to API gateway to be sent to backend lambda function. The lambda function then sends the input pixels to the Sagemaker endpoint.
The endpoint then returns the model's guess. The guess then travels back up the chain to the webapp which then unpacks the return JSON and displays the guess.
![NumberGuesser](https://github.com/PiyushBud/AWS-Number-Guesser/assets/93603829/400a3314-4623-4771-968e-22ccb4c8db8a)
