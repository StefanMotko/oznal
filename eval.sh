echo 'Original data'
python eval.py target_pres.csv predict_pres.csv
echo 'Undersampled data'
python eval.py target_under.csv predict_under.csv
echo 'Oversampled data'
python eval.py target_over.csv predict_over.csv