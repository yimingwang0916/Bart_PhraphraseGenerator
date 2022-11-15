# Bart_PharaphraseGenerator
# Based Bart model from huggingface https://huggingface.co/docs/transformers/model_doc/bart#bart

# Training datasets is pairs of sentences and headlines from newspaper
  to test the code please use train01.json
# Testing datasets is eval.json

#model_device_set.py downloads model and sets device.
#dataset.py gets training data and test data.
#valu_dataset.py gets valuation's data.
#def_train.py defines train and valuation process.
#def_test.py defines test process.
#lv_curve.py draws learning- and valuation-curve.
#ot_curve.py shows BLEU score of original model and after-training model.
#train.py runs training process.
#test.py runs test process.
#timer.py 
