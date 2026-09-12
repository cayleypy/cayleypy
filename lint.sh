result=0

echo "Running black..."
black --check ./cayleypy
result+=$?

echo "Running pylint..."
pylint ./cayleypy
result+=$?


echo "Running pyright..."
pyright ./cayleypy
result+=$?

exit $result
