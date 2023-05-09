import statsmodels.api as sm

def get_linear_model(x_train, y_train, n_samples):
    x_train = x_train[-n_samples:]
    y_train = y_train[-n_samples:]
    x_sm = sm.add_constant(x_train)
    model = sm.OLS(y_train, x_sm).fit()
    return model

def get_pred(model, x_pred, pi_green_light, pi_yellow_light):
    x_pred_sm = sm.add_constant(x_pred)
    y_pred = model.predict(x_pred_sm)
    pred0 = model.get_prediction(x_pred_sm)
    y_pred_green_light = pred0.summary_frame(alpha=(1-pi_green_light))
    y_pred_yellow_light = pred0.summary_frame(alpha=(1-pi_yellow_light))
    return y_pred, y_pred_green_light, y_pred_yellow_light

def get_light_color_class(y_test, green_lower_bound, green_upper_bound, yellow_lower_bound, yellow_upper_bound):
    if green_lower_bound <= y_test <= green_upper_bound:
        return "green"
    elif yellow_lower_bound <= y_test <= yellow_upper_bound:
        return "yellow"
    else:
        return "red"
