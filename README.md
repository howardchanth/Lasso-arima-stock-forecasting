# lasso_stock v0 (Initial Version)

## Hyper-parameters

<table style="width:120%">
  <tr>
    <th colspan = "2">General Parameters</th>
  </tr>
  <tr>
    <th>Prameter</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>START_DATE</td>
    <td>Start date of the data</td>
  </tr>
  <tr>
    <td>END_DATE</td>
    <td>End date of the data</td>
  </tr>
  <tr>
    <td>PRED_END_DATE</td>
    <td>End date of the prediction</td>
  </tr>  
  <tr>
    <td>PRED_DUR</td>
    <td>Duration of prediction. Will be set to PRED_END_DATE - END_DATE if not given (None)</td>
  </tr>
  <tr>
    <td>SERIES_NAME</td>
    <td>Name of the series. Specifically ^HSI for Hang Sang Index, and ^GSPC for S&P 500 index</td>
  </tr>
  <tr>
    <td>IS_REALTIME</td>
    <td>If true, the forecaster will download the data from Yahoo finance, else it will just read the data from current directory using the specified name of the series</td>
  </tr>
  <tr>
    <td>MODEL_NAME</td>
    <td>Model used for prediction. Currently only GBM, ARIMA, Lasso-based ARIMA and Ridge-based ARIMA are available.</td>
  </tr>
  <tr>
    <td>SCENE_SIZE</td>
    <td>Size of scenes when simulating of scenarios</td>
  </tr>  
  <tr>
    <td>ALPHA</td>
    <td>Level of significance of testing</td>
  </tr>
  <tr>
    <td>MODEL_NAME</td>
    <td>Model used for prediction. Currently only GBM, ARIMA, Lasso-based ARIMA and Ridge-based ARIMA are available.</td>
  </tr>
  <tr>
    <td>PLOT_CI</td>
    <td>Whether or not we need to plot the Confidence Interval when plotting the predictions</td>
  </tr>  
  <tr>
    <th colspan = "2">ARIMA Specific Parameters</th>
  </tr>
  <tr>
    <td>d</td>
    <td>Number of differencing needed to make the tie series stationary, equivalent to d in ARIMA(p,d,q) model</th>
  </tr>
  <tr>
    <td>LAG</td>
    <td>Equivalent to p in ARIMA(p,d,q) model, used to determine the order of auto-regressive component in ARIMA</td>
  </tr>
  <tr>
    <td>MA_ORDER</td>
    <td>Equivalent to q in ARIMA(p,d,q) model, used to determine the order of moving average component in ARIMA</td>
  </tr>
</table>

## Sample Run

1. Customize the settings for simulation by adjusting the hyper-parameters in PARAMS
2. Run main.py to obtain the results
