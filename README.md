## Renewable Energy (Wind Energy) Generation Prediction

### Problem Statement
Renewable Energy and Time Series: Renewable energy, especially wind power, generally depends on wind speed and wind direction. Wind speed and direction have a periodic nature (seasonal, daily, weekly, monthly, etc.), and therefore the energy generated by wind becomes a time series problem.

The task at hand is to use windmill turbine data recorded at 10-minute intervals over a period of 30 months to predict wind power.

![alt text](/app/static/image.png)

### Project Summary
This project involves deploying a machine learning model using a Flask web application to predict renewable energy output based on periodic time intervals (minutes, hours, days, months, etc.). The application processes wind turbine data recorded at 10-minute intervals over a period of 30 months to forecast wind power generation. Flask is utilized to create a web interface for users to input data and receive predictions. The model leverages time series analysis to account for the periodic nature of wind speed and direction, ensuring accurate and reliable energy predictions. The web app also serves static files, such as images and results, to enhance the user experience. 

### Tech stacks
```bash
Flask
Pandas
Matplotlib/Seaborn
Scikit-learn
HTML/CSS
