import streamlit as st
import pandas as pd

data = pd.read_excel(r"conclusion.xlsx")
print(data)
def show_report_page():
    st.title("Our Final Report")
    st.image("images/Picture9.png", caption='Would love to have the best insights into the HDB market? Our report will help you to better assess your house\'s values.')

    st.header("1. Data processing")
    st.markdown("""<div style="text-align: justify">The original dataset contains the prices and features of resale HDB flats from 2017 to 2021. Since the inclusion of more data may lower estimation variance and improve the predictive performance of the model, we also experimented with the complete resale HDB price datasets from 1990 to 2022. Both datasets consist of 10 resale HDB flat features, namely the month, town, flat type, block, street name, storey range, floor area, flat model and lease commence date. The newer datasets contain an additional column of ‘remaining lease’. These datasets have both numerical and categorical features. The numerical features are floor area, lease commence date and resale price while the rest are categorical features. Categorical features will be numerically encoded for our regression model. Different encoding methods such as one-hot encoding and target encoding for the categorical features are tested to predict the resale HDB price. </div>""", unsafe_allow_html=True)

    st.subheader("1.1. Data Exploratory Analysis")
    st.markdown("""<div style="text-align: justify">After plotting scatter diagrams to examine the relationship between the features and the resale price, we have decided to drop ‘block’, ‘street_name’. Prior to that, the full address of the HDB is derived by concatenating these two features, which are used to geographically encode our data with latitude and longitude. The coordinates of the HDB allow us to obtain more useful features that are likely to influence buyers’ purchasing decisions such as the distance to the nearest public transport and the nearest shopping mall which will be discussed in the next section.</div>""", unsafe_allow_html=True)

    st.markdown("#### 1.1.1 Month")
    st.markdown("Resale price hardly differs in relation to the months. Interestingly, the frequency of housing is very scarce in April, May 2020 and July 2021. This led to a lower resale price distribution due to the reduced sample size.</div>""", unsafe_allow_html=True)
    st.image("images/Picture1.png", use_column_width=True, caption='Month analysis')

    st.markdown("#### 1.1.2 Town")
    st.markdown("""<div style="text-align: justify">In general, Punggol, Sengkang and Bishan have the top 3 highest resale HDB prices since 1990. Some towns with large resale HDB price ranges include Central area, Bishan and Bukit Timah.</div>""", unsafe_allow_html=True)
    st.image("images/Picture2.png", use_column_width=True, caption='Town analysis')

    st.markdown("#### 1.1.3 Flat type")
    st.markdown("""<div style="text-align: justify">As there is a small sample size of 1 room, 2 room and multi-generation flat types in the dataset, there is insufficient data to make a suitable conclusion from the relationship between flat type and resale price. Since the flat type has a natural order, it will be ordinally encoded.</div>""", unsafe_allow_html=True)
    st.image("images/Picture3.png", use_column_width=True, caption='Flat type analysis')

    st.markdown("#### 1.1.4 Storey range")
    st.markdown("""<div style="text-align: justify">We convert the storey range into a numerical variable by taking the mean of the lower and upper bound of each class. This also preserves the natural ordinal ordering of storey range of the HDB flat. By calculating the median value of the HDB range, the linear regression model yields a slightly better result as the R squared improves by ~0.001. However, the improvement is not guaranteed for another regression model such as the support vector regression.</div>""", unsafe_allow_html=True)
    st.image("images/Picture4.png", use_column_width=True, caption='Storey range analysis')

    st.markdown("#### 1.1.5 Floor area square meter")
    st.markdown("""<div style="text-align: justify">From the 2017-2020 data, a visible trend is that as floor area rises, the resale price is greater. However the same could not be said for the 1990-2022 data.</div>""", unsafe_allow_html=True)
    st.image("images/Picture5.png", use_column_width=True, caption='Floor area square meter analysis')

    st.markdown("#### 1.1.6 Flat model")
    st.markdown("""<div style="text-align: justify">We excluded the following flat model type: Terrace, Improved-Maisonette, Premium Maisonette, Multi Generation, Premium Apartment Loft, 2-room, due to their small sample size. As such, we discover a trend between flat model and resale price. For instance, New Generation, Simplified and Model A2 tend to fetch lower resale price in general.</div>""", unsafe_allow_html=True)
    st.image("images/Picture6.png", use_column_width=True, caption='Flat model analysis')

    st.markdown("#### 1.1.7 Lease Commence Date")
    st.markdown("""<div style="text-align: justify">Lease commence date data was initially presented in years. Generally HDB with later lease commence dates are valued at a greater resale price.</div>""", unsafe_allow_html=True)
    st.image("images/Picture7.png", caption='Lease analysis')

    st.markdown("#### 1.1.8 Resale price")
    st.markdown("""<div style="text-align: justify">We see that the resale HDB price has risen significantly from 1990 to 2019. The property cooling measures introduced in 2013 and 2018 have led to a slight dip in the resale HDB prices.</div>""", unsafe_allow_html=True)
    st.image("images/Picture8.png", use_column_width=True, caption='Resale price analysis')

    st.subheader("1.2 Feature Synthesis")
    st.markdown("""<div style="text-align: justify">We also consider other factors that influence the HDB resale price, such as accessibility to public transport, shopping and lifestyle amenities and maturity of the town. Therefore, we have obtained relevant data to increase the accuracy of the model.</div>""", unsafe_allow_html=True)


    st.markdown("#### 1.2.1 Proximity to MRT")
    st.markdown("""<div style="text-align: justify">The minimum distance to MRT of the resale HDB is obtained by geodesic measure with the coordinates of the HDB address and the coordinates of the list of MRT stations. The list of MRT stations is scraped from MRT Map SG. The coordinates of the MRT and HDB address are acquired with the Onemaps API.</div>""", unsafe_allow_html=True)

    st.markdown("#### 1.2.2 Proximity to Commercial Buildings")
    st.markdown("""<div style="text-align: justify">The proximity to the nearest shopping mall is obtained with a similar method as to how the proximity to MRT is calculated.</div>""", unsafe_allow_html=True)
    st.image("images/Picture9.png", use_column_width=True, caption='Spatial features analysis')

    st.markdown("#### 1.2.3 Availability of Markets or Hawker Centers")
    st.markdown("""<div style="text-align: justify">The availability of markets or hawker centers of the resale HDB is retrieved from the HDB property information dataset. By mapping the addresses of the resale HDB price dataset, the presence of market or hawker centers is binarily encoded.</div>""", unsafe_allow_html=True)


    st.markdown("#### 1.2.4 Town Maturity")
    st.markdown("""<div style="text-align: justify">The list of mature towns is obtained from various house property websites. Mature towns usually have a more extensive network of public transport and more developed facilities. Town maturity is included as a predictor as it can imply the convenience and accessibility of the HDB flats, which affects buyers’ purchasing decisions. However, there is no clear explanation in the classification of town maturity. For instance, towns like Punggol and Sengkang, which have well-established public transport and amenities can be listed as non-mature. The town maturity is also binarily encoded.</div>""", unsafe_allow_html=True)

    st.markdown("#### 1.2.5 Region")
    st.markdown("""<div style="text-align: justify">Since there are a total of 27 towns, one hot encoding would lead to an increase of 27 dimensions and sparsity of the data. The performance of the model deteriorates with higher dimensions due to the curse of dimensionality. Hence, towns can be grouped into their respective regions (central, east, north, northeast or west regions) to reduce the number of dimensions of the model.</div>""", unsafe_allow_html=True)

    st.subheader("1.3 Feature Engineering")

    st.markdown("#### 1.3.1 Multicollinearity")
    st.markdown("""<div style="text-align: justify">Multicollinearity occurs when two or more independent variables are highly correlated with one another in a regression model. It can pose an issue in interpreting the impact of one feature on its HDB resale price. However, multicollinearity may not affect the model’s predictive accuracy.</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify">To check the collinearity between categorical variables, they must first be converted into numerical variables. This can be done with target encoding, where the categorical value is substituted with the mean of the resale price. Next, multicollinearity between predictors can be examined with variance inflation factor (VIF) or correlation matrix. In this case, we see a high correlation between floor area and flat type. Thus, one of the variables with high collinearity may be dropped to improve the model’s interpretability.</div>""", unsafe_allow_html=True)
       
    st.markdown("#### 1.3.2 Selected Features")
    st.markdown("""<div style="text-align: justify">There are a total of 12 selected features used for our model. The features are as follows: ‘month’, ‘flat_type’, ‘floor_area_sqm’, ‘flat_model’, ‘lease_commence_date’,’ availability of market/hawker’, ‘town maturity’, ‘region’, ‘distance to the nearest MRT station’, ‘distance to the nearest shopping mall’ and ‘storey_range’.""", unsafe_allow_html=True)

    st.markdown("#### 1.3.3 Feature Scaling")
    st.markdown("""<div style="text-align: justify">From the distribution plot of the resale HDB price, we observe that the resale HDB price is rightly skewed. Therefore, we perform log transformation on the resale HDB price before passing it into the model as the linear regression model assumes that the residuals are normally distributed. Though feature scaling on the independent variables has minimal impact on predictive accuracy, we decided to standardize the coefficients to better interpret the effect each predictor has on resale HDB price.</div>""", unsafe_allow_html=True)
    
    st.subheader("2. Model")
    st.markdown("""<div style="text-align: justify">We have implemented different regression models to predict the resale HDB price. The regression models are run with the same features as mentioned above. The deep learning approach is also used to compare against the performance of regression models.</div>""", unsafe_allow_html=True)

    st.subheader("2.1 Linear Regression")
    st.markdown("""<div style="text-align: justify">Linear regression serves as the baseline of our models. The performance metric of the model is ordinary least squares. Considering that the trend of resale HDB price might be cyclical, we split the testing and training data into different periods and compare the performance of the model. Testing with 6, 12, 36, and 60 periods respectively, the best R square for prediction is 0.9372 where the 60th month is used for testing and the previous 59 months are reserved for training. Therefore, we set the test period to be 60 in our subsequent experimentation. As the R square of the model is lower than the prediction R square, we did not regularize the model.</div>""", unsafe_allow_html=True)
    st.image("images/Picture10.png", use_column_width=True, caption='R2 Comparision')

    st.subheader("2.2 Polynomial Regression")
    st.markdown("""<div style="text-align: justify">Polynomial regression is used to examine if the variables and resale HDB price have a nonlinear relationship. The R2 of the model has improved while the R2 for prediction has decreased drastically. This may be caused by overfitting. Hence, we test ridge regression with polynomial features to address overfitting. </div>""", unsafe_allow_html=True)

    st.subheader("2.3 Ridge Regression")
    st.markdown("""<div style="text-align: justify">Ridge regression introduces a regularization parameter to penalize overfitting in a complicated model. The parameter alpha in ridge regression which determines the amount of regularization is tuned to find the optimal one for our model. To optimize the regularisation parameter, an exhaustive search with GridSearchCV is used. The best model yields a predictive R2 of 0.945, where optimal alpha is 2.5.</div>""", unsafe_allow_html=True)

    st.subheader("2.4 Support Vector Regression")
    st.markdown("""<div style="text-align: justify">For support vector regression, we set epsilon to be 0 and test the regularization parameter with 1, 10, and 50. The test period is set to 60. The maximum iteration is set to 1000000 to ensure the algorithm reaches the optimal solution. The best model yields a predictive R2 of 0.9109 with a parameter equal to 10.</div>""", unsafe_allow_html=True)

    st.subheader("2.5 Multi-Layer Perceptron Model (Deep Learning)")
    st.markdown("""<div style="text-align: justify">MLP Model is used to compare with the linear models. It is trained using the same data with similar features, so it will have 34 nodes in the input layer. Before training, we scaled the data using the StandardScaler to make the model converge faster and more stable. The first version of the model comprises 4 layers with the largest layer having 200 nodes. We trained it for 500 epochs without any dropout layer. The R2 of the model is 0.979699 and the R2 for predictions is  0.9783296 with an RMSE of 23467.35616 (without log transform).</div>""", unsafe_allow_html=True)
    st.image("images/Picture11.png", use_column_width=True, caption='Deep learning loss graph')

    st.markdown("""<div style="text-align: justify">The loss decreases rapidly after 10-15 epochs, then the model converges slowly. It can be observed that the loss is still going down when we stop the training, which means it can be reduced further if we increase the number of epochs. We also tried the model with more layers and more nodes, details are in the appendix.</div>""", unsafe_allow_html=True)

    st.header("3. Conclusion")

    st.subheader("3.1 Analysis of Regression Model")
    st.markdown("""<div style="text-align: justify">In general, the features with the greatest influence tend to be town maturity, flat type, flat model, and some towns like Hougang and Yishun. However, the coefficients of the independent variables change according to different scaling methods, models, and parameters. Therefore, it is challenging to pinpoint the impact each variable has on the resale HDB price</div>""", unsafe_allow_html=True)
    st.table(data)
    st.markdown("""<div style="text-align: justify">Models with the best performance after tuning the parameters are listed for further comparison.</div>""", unsafe_allow_html=True)

    st.subheader("3.2 Evaluation of Model")
    st.markdown("""<div style="text-align: justify">Utilizing training data and test data, we discover that the ridge regression polynomial model gave the best R2 prediction, with a close second of linear regression.</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify">Deep learning models are proven to be more efficient in learning the data, hence, giving more accurate predictions. The MLP model we used gave better R2 statistics.</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify">Given that we have a large dataset with just 34 features (after encoding), the performance of the ridge regression model with polynomial (linear regression) is mostly as good as the deep learning model, evident by their close R2 statistic of prediction. Overall, most of our models perform better than the model illustrated in the class worksheet example after additional feature engineering and tuning of parameters.</div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: justify">To improve the model, additional feature engineering can be conducted to remove variables with high collinearity and include other useful features. Further tuning of parameters in both regression and deep learning models can also be done to optimize model performance. Finally, cross-validation can be performed to select the best model amongst the models that have been proposed. </div>""", unsafe_allow_html=True)

    # st.subheader("3.1 Analysis of Regression Model")
    # st.markdown("""<div style="text-align: justify"></div>""", unsafe_allow_html=True)