# Team Members:
- Mohammed Alaklabi
- Faisal Alossaimi
- Saud Alotaibi
- Rand Alharbi
- Abdullah Altuwayjiri


# Streamlit:
[ https://use-case-8-project-5-ndyrl9snwivjwtifgjychx.streamlit.app ]

# Deployment
[ https://use-case-8-project-5.onrender.com ]
## Introduction:

Restaurant ratings serve as a valuable reference for both consumers and restaurants.
Restaurant ratings influence how much money a restaurant makes and help customers choose where to eat.

## Dataset Synopsis and Origin:
 ![alt text](image.png)
 ##### data before cleaning

## Model Selection:
- DBSCAN
- K-Means

## Feature Engineering:
- separate the column Types into multiple columns
- Weighted Rating (Rating*Number of Ratings)

![alt text](image-1.png)

## Performance Metric Visuals:
DBSCAN
![alt text](image-2.png)

K-Means
![alt text](image-3.png)


![alt text](newplot.png)

## Best Model Determination:
### DBSCAN
appears to be better at capturing the underlying structure of the data, considering it identifies noise and does not force every point into a cluster.


## Feature and Prediction Insights:
importent Features 
Here is the modified text:

- Rating  
- Number of Ratings   
- bakery  
- Weighted Rating  
- Name Al Jalab Restaurant  
- Name Al Nafoura Restaurant  
- Name Al Romansiah  
- Name Dining Hall  
- Name Fairmont Riyadh  
- Name Hardee's  
- Name Herfy  
- Name KFC  
- Name Mama Noura  
- Name Sheikh Al-Mandi Restaurant  
- Name TOKYO  
- Name Wooden Bakery  
- Name مركز ماي كار لصيانة السيارات الاوروبية  
- Name مطعم السحاب  
- Name مطعم ندى مطلق العتيبي  
- Name وايت قاردن  
- neighborhood حي القدس  

