result=0
pylint ./cayleypy
result+=$?
mypy ./cayleypy
result+=$?
exit $result
