# Makefile to build the hardware topology C extension

CC = gcc
CFLAGS = -shared -fPIC -O3
TARGET = hardware/libtopology.so
SRC = hardware/topology_field.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
