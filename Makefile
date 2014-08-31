
CXXFLAGS=`pkg-config --cflags --libs opencv`

all: squares webcam

squares: squares.cpp
webcam: webcam.cpp

