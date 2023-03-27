import statsmodels.api as sm

def get_linear_model(x_train, y_train, n_samples):
    x_train = x_train[-n_samples:]
    y_train = y_train[-n_samples:]
    x_sm = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_sm).fit()
    return model

def get_pred(model, x_pred):
    x_pred_sm = sm.add_constant(x_pred)
    y_pred = model.predict(x_pred_sm)
    pred0 = model.get_prediction(x_pred_sm)
    y_pred_80 = pred0.summary_frame(alpha=0.2)
    y_pred_95 = pred0.summary_frame(alpha=0.05)
    return y_pred, y_pred_80, y_pred_95
