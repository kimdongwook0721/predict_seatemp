<h1>Building an emulator for sea surface temperature</h1>

<h1>Abstract</h1>
<h4>We are interested in the problem of predicting sea surface temperature with machine learning</h4>
<h4>methods using only the relevant climate factors. We tried to resolve the cost inefficiency of</h4>
<h4>the temperature predicting model uses over a million of data[1] that is observed by about 6300[2]</h4>
<h4>NASA weather stations, a typical model that prominent agencies like NASA uses. First of all, we</h4>
<h4>minimized the required computing power by using only velocity and pressure variables.</h4>
<h4>Moreover, we minimized the usage of data by using only 14G of NASA-CISS data and</h4>
<h4>supplemented the minimized data by comparing four different machine learning models to</h4>
<h4>predict global sea surface temperature. In addition, beyond predicting the global sea surface</h4>
<h4>temperature, we also wanted to delve into predicting each region’s temperature. Thus, we</h4>
<h4>gridded the earth into 64,800 regions, based on latitudes and longitudes. However, the accuracy</h4>
<h4>drastically decreased since the data was also divided into 64,800 parts. Therefore, we applied the</h4>
<h4>Gaussian filter to our predicted outcomes, and it made a breakthrough in accuracy. This will have</h4>
<h4>a huge impact on temperature prediction because it will astoundingly decrease the expense and</h4>
<h4>the amount of data needed for predicting temperature.</h4>
