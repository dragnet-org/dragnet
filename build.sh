find . -name "*.pyx" | xargs cython --cplus && \
echo "\nBuilt modules:" && \
find . -name *.cpp
