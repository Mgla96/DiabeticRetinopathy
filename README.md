# Diabetic Retinopathy
This program predicts whether a patient has a medical condition known as Diabetic Retinopathy based on both categorical(integer) and continuous(float) features. This was created for learning purposes only. I implemented a decision tree machine learning model with random forests to increase accuracy and threshold splits to prevent overfitting.


These are the 18 features used below: <br/>

### 0
The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.<br/>
### 1
The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack. <br/>
### 2-7
The results of MA detection. Each feature value stand for the number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.<br/>
### 8-15
contain the same information as 2-7) for exudates. However,as exudates are represented by a set of points rather than the number of pixels constructing the lesions, these features are normalized by dividing the number of lesions with the diameter of the ROI to compensate different image sizes.<br/>
### 16
The euclidean distance of the center of the macula and the center of the optic disc to provide important information regarding the patient's condition. This feature is also normalized with the diameter of the ROI.<br/>
### 17
The diameter of the optic disc.<br/>
### 18
The binary result of the AM/FM-based classification. <br/>

