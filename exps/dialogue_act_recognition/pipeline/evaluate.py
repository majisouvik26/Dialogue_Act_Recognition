from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def evaluate_model(model, X_test, y_test, is_deep_model=False):
    if is_deep_model:
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return acc, report
