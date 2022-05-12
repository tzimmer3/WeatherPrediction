import pandas as pd


########################################
# General Utilities
########################################

def set_date_index(data):
    """
    Set the index to the date column. Required: Date column is called ['Date']

    Inputs:
        Data - DataFrame with a Date column.

    Outputs:
        Data - DataFrame with Date column as the index.
    """
    data = data.set_index('Date')
    return data


def datetime_datatype(data):
    """
    Convert a column to datetime. Required: Date column is called ['Date']

    Inputs:
        Data - DataFrame with a Date column.

    Outputs:
        Data - DataFrame with Date column cast as to_datetime.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    return data

########################################
# Date Slicing
########################################

def date_slicer_start(data, start_date):
    """
    Ability to slice data from a specific start date forward.  Required: Date column is called ['Date']

    Inputs:
        Data - DataFrame with a Date column.
        Start Date - Date of demarkation.  Observations with prior dates will be removed.  Observations after this date will remain.

    Outputs:
        Sliced DataFrame - DataFrame with observations after start date.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    sliced_df = data[~(data['Date'] < start_date)]
    return sliced_df


def date_slicer_end(data, end_date):
    """
    Ability to slice data from a specific start date backward.  Required: Date column is called ['Date']

    Inputs:
        Data - DataFrame with a Date column.
        End Date - Date of demarkation.  Observations with prior dates will remain.  Observations after this date will be removed.

    Outputs:
        Sliced DataFrame - DataFrame with observations before start date.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    sliced_df = data[(data['Date'] < end_date)]
    return sliced_df

# Ability to slice data to focus on specific window
def date_slicer_window(data, start_date, end_date):
    """
    Ability to slice data to focus on specific window.  Required: Date column is called ['Date']

    Inputs:
        Data - DataFrame with a Date column.
        Start Date - Date of demarkation to start the window.
        End Date - Date of demarkation to end the window.

    Outputs:
        Sliced DataFrame - DataFrame with observations within specified window.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    sliced_df = data[~(data['Date'] < start_date) & (data['Date'] < end_date)]
    return sliced_df

def date_slicer_range(data, start_date, integer):
    """
    Ability to slice data to focus on specific window.  Required: Date column is called ['Date']

    Inputs:
        Data - DataFrame with a Date column.
        Start Date - Date of demarkation to start the window.
        Integer - Number of timesteps to keep in the data past the Start Date.

    Outputs:
        Sliced DataFrame - DataFrame with observations within specified window.
    """
    pass
    end_date = start_date + integer
    data['Date'] = pd.to_datetime(data['Date'])
    sliced_df = data[~(data['Date'] < start_date) & (data['Date'] < end_date)]
    return sliced_df



########################################
# Date Based Train/Test Split
########################################

# Split to train/test using a point in time instead of random sampling
def train_test_split_dates(X,y,split_date):
    """
    Split to train/test using a point in time instead of random sampling

    Inputs:
        X - DataFrame with independent columns
        y - DataFrame with dependent column
        Split_Date - Date of demarkation.  Data prior to this date will be train.  Data after this date will be test.

    Outputs:
        Predicted Dates - List
    """
    # Recast as datetime
    X['Date'] = pd.to_datetime(X['Date'])
    y['Date'] = pd.to_datetime(y['Date'])
    # Training set
    X_train = X[(X['Date'] < split_date)]
    y_train = y[(y['Date'] < split_date)]
    # Test set
    X_test = X[~(X['Date'] < split_date)]
    y_test = y[~(y['Date'] < split_date)]
    return X_train, X_test, y_train, y_test

###########################################
# Remove Date Column
###########################################

def remove_date_column(X_train, X_test, y_train, y_test):
    """
    Removes the date column from DataFrame prior to modeling.  To be used right after train/test split.

    Inputs:
        X_train dataframe
        X_test dataframe
        y_train dataframe
        y_test dataframe

    Outputs:
        Same as inputs.  Less the Date column in all DataFrames.
    """
    del X_train['Date']
    del y_train['Date']
    del X_test['Date']
    del y_test['Date']
    return X_train, X_test, y_train, y_test


###########################################
# Create Date Lists for Prediction Dataset
###########################################

def create_dates_days(data, points_to_predict):
    """
    [Day] Build a list of dates to merge with predictions array

    Inputs:
        Data - Training dataset.  Last date in training set is identified.
        Points to predict - Number of observations in prediction dataset.

    Outputs:
        Predicted Dates - List
    """
    last_date = data.index[-1]
    predicted_dates = []
    for _ in range(points_to_predict):
        next_date = last_date + relativedelta(days=1)
        predicted_dates.append(next_date)
        last_date = next_date
    return predicted_dates


def create_dates_weeks(data, points_to_predict):
    """
    [Week] Build a list of dates to merge with predictions array

    Inputs:
        Data - Training dataset.  Last date in training set is identified.
        Points to predict - Number of observations in prediction dataset.

    Outputs:
        Predicted Dates - List
    """
    last_date = data.index[-1]
    predicted_dates = []
    for _ in range(points_to_predict):
        next_date = last_date + relativedelta(weeks=1)
        predicted_dates.append(next_date)
        last_date = next_date
    return predicted_dates


def create_dates_months(data, points_to_predict):
    """
    [Month] Build a list of dates to merge with predictions array

    Inputs:
        Data - Training dataset.  Last date in training set is identified.
        Points to predict - Number of observations in prediction dataset.

    Outputs:
        Predicted Dates - List
    """
    predicted_dates = []
    last_date = data.index[-1]
    for _ in range(points_to_predict):
        next_date = last_date + relativedelta(months=1)
        predicted_dates.append(next_date)
        last_date = next_date
    return predicted_dates

###########################################
# Prediction DataFrame Construction
###########################################

def df_constructor_predictions(date, predicted):
    """
    Construct the prediction array as DF with Date column.

    Inputs:
        Date - List - List of dates generated in create_days function.
        Predicted - Array - Predicted values from the model.

    Outputs:
        Predictions - DataFrame with two columns: Date, Prediction.
    """
    predictions = pd.DataFrame(date)
    predictions['predicted'] = predicted
    predictions.columns = ['Date','Prediction']
    return predictions