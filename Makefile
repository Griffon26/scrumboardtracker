
CXXFLAGS=`pkg-config --cflags --libs opencv`

all: squares

squares: squares.cpp

