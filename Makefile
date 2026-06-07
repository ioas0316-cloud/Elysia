# Makefile to build the hardware topology C extension

CC = gcc
CFLAGS = -shared -fPIC -O3
TARGET = core/hardware/libtopology.so
SRC = core/hardware/topology_field.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
