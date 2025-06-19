result=0

echo "Running pylint..."
pylint ./cayleypy
result+=$?

echo "Running mypy..."
mypy ./cayleypy
result+=$?

exit $result
