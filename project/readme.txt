1. generate_model.py
running in command: 
python generate_model.py filename

python generate_model.py iris.txt.shuffled
result: you can find model.txt generated

2. classifier.py
running in command:  
python classifier.py model.txt test.txt 
or python classifier.py model.txt iris.txt.shuffled

result: 
The data with predicted class will be output and
you can find output.txt generated

3.cross_validation.py
running in command: 
python cross_validation.py iris.txt.shuffled

result: you can find confusion matrixes and precision calculated from the 3- fold cross validation