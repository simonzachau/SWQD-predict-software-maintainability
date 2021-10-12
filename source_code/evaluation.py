from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, mean_squared_error, classification_report

def evaluate(y_pred, y_true):
	results = [
		'prediction:',
		y_pred,
		'truth:',
		y_true
	]

	
	unpredicted_classes = set(y_true) - set(y_pred)
	if len(unpredicted_classes) > 0:
		results.extend([
			'classes that were not predicted:',
			unpredicted_classes,
			'Classification report only for predicted classes:',
			classification_report(y_true, y_pred, labels=np.unique(y_pred), digits=4)
		])

	results.extend([
		'Classification report for all classes:',
		classification_report(y_true, y_pred, digits=4),
		'Matthews Correlation Coefficient: %.3f' % matthews_corrcoef(y_true, y_pred),
		'Cohen’s Kappa Score: %.3f' % cohen_kappa_score(y_true, y_pred),
		'Mean Squared Error: %.3f' % mean_squared_error(y_true, y_pred)
	])

	return results
